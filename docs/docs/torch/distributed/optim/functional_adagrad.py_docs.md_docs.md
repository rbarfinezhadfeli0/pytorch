# Documentation: `docs/torch/distributed/optim/functional_adagrad.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/optim/functional_adagrad.py_docs.md`
- **Size**: 6,979 bytes (6.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/optim/functional_adagrad.py`

## File Metadata

- **Path**: `torch/distributed/optim/functional_adagrad.py`
- **Size**: 4,304 bytes (4.20 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Optional

import torch
import torch.optim._functional as F
from torch import Tensor
from torch.distributed.optim._deprecation_warning import (
    _scripted_functional_optimizer_deprecation_warning,
)


__all__: list[str] = []


# Define a TorchScript compatible Functional Adagrad Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly let the user pass gradients to the `step` function
# this is so that we could separate the gradients and parameters
# and allow multithreaded trainer to update the parameters
# without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
@torch.jit.script
class _FunctionalAdagrad:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        warmup_lr_multiplier: float = 1.0,
        warmup_num_iters: float = 0.0,
        eps: float = 1e-10,
        coalesce_grad: bool = True,
        foreach: bool = False,
        fused: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        self.defaults = {
            "lr": lr,
            "lr_decay": lr_decay,
            "eps": eps,
            "weight_decay": weight_decay,
            "initial_accumulator_value": initial_accumulator_value,
            "warmup_lr_multiplier": warmup_lr_multiplier,
            "warmup_num_iters": warmup_num_iters,
        }
        self.coalesce_grad = coalesce_grad
        self.foreach = foreach
        self.fused = fused
        self.maximize = maximize
        self.state = torch.jit.annotate(dict[torch.Tensor, dict[str, torch.Tensor]], {})

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

        # TODO: no union or any types in TorchScript, make step a scalar tensor instead
        # This is also needed by if we want to share_memory on the step across processes
        for p in self.param_group["params"]:
            self.state[p] = {
                "sum": torch.full_like(p.data, initial_accumulator_value),
                "step": torch.tensor(0.0),
            }

    def step(self, gradients: list[Optional[Tensor]]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        state_sums = []
        state_steps: list[Tensor] = []

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        has_sparse_grad, has_complex = False, False
        for param, gradient in zip(self.param_group["params"], gradients):
            if gradient is not None:
                has_sparse_grad |= gradient.is_sparse
                has_complex |= torch.is_complex(param)
                params_with_grad.append(param)
                grads.append(gradient)
                state = self.state[param]
                state_sums.append(state["sum"])
                state_steps.append(state["step"])

        with torch.no_grad():
            F.adagrad(
                params,
                grads,
                state_sums,
                state_steps,
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                lr_decay=self.defaults["lr_decay"],
                eps=self.defaults["eps"],
                has_sparse_grad=has_sparse_grad,
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
            )

```



## High-Level Overview


This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_FunctionalAdagrad`

**Functions defined**: `__init__`, `step`

**Key imports**: Optional, torch, torch.optim._functional as F, Tensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/optim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional
- `torch`
- `torch.optim._functional as F`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`torch/distributed/optim`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`zero_redundancy_optimizer.pyi_docs.md`](./zero_redundancy_optimizer.pyi_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`functional_adadelta.py_docs.md`](./functional_adadelta.py_docs.md)
- [`post_localSGD_optimizer.py_docs.md`](./post_localSGD_optimizer.py_docs.md)
- [`functional_adamax.py_docs.md`](./functional_adamax.py_docs.md)
- [`named_optimizer.py_docs.md`](./named_optimizer.py_docs.md)
- [`functional_rprop.py_docs.md`](./functional_rprop.py_docs.md)
- [`functional_adam.py_docs.md`](./functional_adam.py_docs.md)


## Cross-References

- **File Documentation**: `functional_adagrad.py_docs.md`
- **Keyword Index**: `functional_adagrad.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/optim`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/optim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/distributed/optim`):

- [`apply_optimizer_in_backward.py_docs.md_docs.md`](./apply_optimizer_in_backward.py_docs.md_docs.md)
- [`functional_rprop.py_kw.md_docs.md`](./functional_rprop.py_kw.md_docs.md)
- [`zero_redundancy_optimizer.py_docs.md_docs.md`](./zero_redundancy_optimizer.py_docs.md_docs.md)
- [`_deprecation_warning.py_kw.md_docs.md`](./_deprecation_warning.py_kw.md_docs.md)
- [`zero_redundancy_optimizer.py_kw.md_docs.md`](./zero_redundancy_optimizer.py_kw.md_docs.md)
- [`functional_rmsprop.py_docs.md_docs.md`](./functional_rmsprop.py_docs.md_docs.md)
- [`functional_rprop.py_docs.md_docs.md`](./functional_rprop.py_docs.md_docs.md)
- [`named_optimizer.py_docs.md_docs.md`](./named_optimizer.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `functional_adagrad.py_docs.md_docs.md`
- **Keyword Index**: `functional_adagrad.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
