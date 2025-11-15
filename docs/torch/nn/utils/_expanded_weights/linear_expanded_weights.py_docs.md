# Documentation: `torch/nn/utils/_expanded_weights/linear_expanded_weights.py`

## File Metadata

- **Path**: `torch/nn/utils/_expanded_weights/linear_expanded_weights.py`
- **Size**: 2,259 bytes (2.21 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs

import torch
import torch.nn.functional as F

from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import (
    forward_helper,
    is_batch_first,
    set_grad_sample_if_exists,
    unpack_expanded_weight_or_tensor,
)


@implements_per_sample_grads(F.linear)
class LinearPerSampleGrad(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, _, __, *expanded_args_and_kwargs):
        if len(expanded_args_and_kwargs[0].shape) <= 1:
            raise RuntimeError(
                "Input does not have a batch dimension. Expanded Weights expected input "
                f"of at least rank 2, got of rank {len(expanded_args_and_kwargs[0].shape)}"
            )
        expanded_kwargs = {
            "bias": expanded_args_and_kwargs[2]
            if len(expanded_args_and_kwargs) == 3
            else None
        }
        expanded_args = expanded_args_and_kwargs[:2]
        ctx.batch_first = is_batch_first(expanded_args_and_kwargs)
        output = forward_helper(F.linear, expanded_args, expanded_kwargs)
        ctx.args = expanded_args
        ctx.kwargs = expanded_kwargs
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        input, weight = ctx.args
        bias = ctx.kwargs["bias"]
        results: list[torch.Tensor | None] = []
        results.append(None)  # for kwarg_names
        results.append(None)  # for op reference

        if input.requires_grad:
            results.append(grad_output.matmul(unpack_expanded_weight_or_tensor(weight)))
        else:
            results.append(None)
        results.extend([None] * 2)  # weight and bias don't compute batched gradients

        if not ctx.batch_first:
            grad_output = grad_output.transpose(0, 1)
            input = input.transpose(0, 1)

        # weight and bias get their grad_sample fields set directly if they exist
        set_grad_sample_if_exists(
            weight, lambda _: torch.einsum("n...i,n...j->nij", grad_output, input)
        )
        set_grad_sample_if_exists(
            bias, lambda _: torch.einsum("n...k->nk", grad_output)
        )
        return tuple(results)

```



## High-Level Overview


This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LinearPerSampleGrad`

**Functions defined**: `forward`, `backward`

**Key imports**: torch, torch.nn.functional as F, implements_per_sample_grads


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/utils/_expanded_weights`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.nn.functional as F`
- `.expanded_weights_impl`: implements_per_sample_grads


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
- [`expanded_weights_utils.py_docs.md`](./expanded_weights_utils.py_docs.md)
- [`expanded_weights_impl.py_docs.md`](./expanded_weights_impl.py_docs.md)
- [`instance_norm_expanded_weights.py_docs.md`](./instance_norm_expanded_weights.py_docs.md)
- [`conv_utils.py_docs.md`](./conv_utils.py_docs.md)
- [`conv_expanded_weights.py_docs.md`](./conv_expanded_weights.py_docs.md)
- [`embedding_expanded_weights.py_docs.md`](./embedding_expanded_weights.py_docs.md)


## Cross-References

- **File Documentation**: `linear_expanded_weights.py_docs.md`
- **Keyword Index**: `linear_expanded_weights.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
