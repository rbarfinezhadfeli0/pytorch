# Documentation: `docs/torch/nn/utils/_expanded_weights/embedding_expanded_weights.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/utils/_expanded_weights/embedding_expanded_weights.py_docs.md`
- **Size**: 5,946 bytes (5.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/utils/_expanded_weights/embedding_expanded_weights.py`

## File Metadata

- **Path**: `torch/nn/utils/_expanded_weights/embedding_expanded_weights.py`
- **Size**: 3,073 bytes (3.00 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from typing import Any

import torch
import torch.nn.functional as F

from .expanded_weights_impl import implements_per_sample_grads
from .expanded_weights_utils import (
    forward_helper,
    set_grad_sample_if_exists,
    standard_kwargs,
)


@implements_per_sample_grads(F.embedding)
class EmbeddingPerSampleGrad(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx: Any, kwarg_names: list[str], _: Any, *expanded_args_and_kwargs: Any
    ) -> torch.Tensor:
        expanded_args, expanded_kwargs = standard_kwargs(
            kwarg_names, expanded_args_and_kwargs
        )
        if len(expanded_args[0].shape) == 1:
            raise RuntimeError(
                f"Expanded Weights needs an input with a batch size, got a 1D tensor, {expanded_args[0]}"
            )
        output = forward_helper(F.embedding, expanded_args, expanded_kwargs)
        ctx.input, ctx.weight = expanded_args
        ctx.padding_idx, ctx.scale_grad_by_freq = (
            expanded_kwargs["padding_idx"],
            expanded_kwargs["scale_grad_by_freq"],
        )
        ctx.sparse = expanded_kwargs["sparse"]
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        input, weight = ctx.input, ctx.weight
        padding_idx, scale_grad_by_freq, sparse = (
            ctx.padding_idx,
            ctx.scale_grad_by_freq,
            ctx.sparse,
        )

        def weight_per_sample_grad(weight: torch.Tensor) -> torch.Tensor:
            batch_size = input.shape[0]
            embedding_dim = weight.shape[1]
            index = (
                input.unsqueeze(-1)
                .expand(*input.shape, embedding_dim)
                .reshape(batch_size, -1, embedding_dim)
            )
            grad_sample = torch.zeros(  # type: ignore[attr-defined]
                batch_size, *weight.shape, device=weight.device, dtype=grad_output.dtype
            )
            return grad_sample.scatter_add_(
                1, index, grad_output.reshape(batch_size, -1, embedding_dim)
            )

        results: list[torch.Tensor | None] = []
        results.append(None)  # for kwarg names
        results.append(None)  # for op reference

        if input.requires_grad:
            bw_fn = torch.ops.aten.embedding_backward
            results.append(
                bw_fn(
                    grad_output,
                    input,
                    weight.shape[0],
                    padding_idx,
                    scale_grad_by_freq,
                    sparse,
                )
            )
        else:
            results.append(None)

        # weight doesn't compute batched gradients; no other arguments are differentiable (2 not saved from forward)
        results = results + [None] * 6

        # set grad_sample field for weight with per sample gradients
        set_grad_sample_if_exists(weight, weight_per_sample_grad)
        return tuple(results)

```



## High-Level Overview


This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `EmbeddingPerSampleGrad`

**Functions defined**: `forward`, `backward`, `weight_per_sample_grad`

**Key imports**: Any, torch, torch.nn.functional as F, implements_per_sample_grads


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/utils/_expanded_weights`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
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
- [`linear_expanded_weights.py_docs.md`](./linear_expanded_weights.py_docs.md)
- [`expanded_weights_utils.py_docs.md`](./expanded_weights_utils.py_docs.md)
- [`expanded_weights_impl.py_docs.md`](./expanded_weights_impl.py_docs.md)
- [`instance_norm_expanded_weights.py_docs.md`](./instance_norm_expanded_weights.py_docs.md)
- [`conv_utils.py_docs.md`](./conv_utils.py_docs.md)
- [`conv_expanded_weights.py_docs.md`](./conv_expanded_weights.py_docs.md)


## Cross-References

- **File Documentation**: `embedding_expanded_weights.py_docs.md`
- **Keyword Index**: `embedding_expanded_weights.py_kw.md`
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
- [`expanded_weights_impl.py_kw.md_docs.md`](./expanded_weights_impl.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`linear_expanded_weights.py_kw.md_docs.md`](./linear_expanded_weights.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `embedding_expanded_weights.py_docs.md_docs.md`
- **Keyword Index**: `embedding_expanded_weights.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
