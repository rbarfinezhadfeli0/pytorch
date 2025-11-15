# Documentation: `docs/torch/distributed/optim/functional_adamw.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/optim/functional_adamw.py_docs.md`
- **Size**: 10,293 bytes (10.05 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributed/optim/functional_adamw.py`

## File Metadata

- **Path**: `torch/distributed/optim/functional_adamw.py`
- **Size**: 7,545 bytes (7.37 KB)
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


# Define a TorchScript compatible Functional AdamW Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
@torch.jit.script
class _FunctionalAdamW:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: bool = False,
        fused: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.defaults = {
            "lr": lr,
            "eps": eps,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
        }
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.foreach = foreach
        self.fused = fused
        self.state = torch.jit.annotate(dict[torch.Tensor, dict[str, torch.Tensor]], {})

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

    def step_param(self, param: Tensor, grad: Optional[Tensor]):
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps: list[Tensor] = []
        has_complex = torch.is_complex(param)
        if grad is not None:
            params_with_grad.append(param)
            grads.append(grad)
        # Lazy state initialization
        if param not in self.state:
            self.state[param] = {}
            state = self.state[param]
            state["step"] = torch.tensor(0.0)
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
            # Exponential moving average of squared gradient values
            state["exp_avg_sq"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
            if self.amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state["max_exp_avg_sq"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )

        state = self.state[param]

        exp_avgs.append(state["exp_avg"])
        exp_avg_sqs.append(state["exp_avg_sq"])

        if self.amsgrad:
            max_exp_avg_sqs.append(state["max_exp_avg_sq"])

        state_steps.append(state["step"])
        with torch.no_grad():
            F.adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=self.amsgrad,
                maximize=self.maximize,
                beta1=self.defaults["beta1"],
                beta2=self.defaults["beta2"],
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                eps=self.defaults["eps"],
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
                has_complex=has_complex,
            )

    def step(self, gradients: list[Optional[Tensor]]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps: list[Tensor] = []

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        has_complex = False
        for param, gradient in zip(self.param_group["params"], gradients):
            if gradient is not None:
                has_complex |= torch.is_complex(param)
                params_with_grad.append(param)
                grads.append(gradient)
                # Lazy state initialization
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state["step"] = torch.tensor(0.0)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    if self.amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )

                state = self.state[param]

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if self.amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                state_steps.append(state["step"])

        with torch.no_grad():
            F.adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=self.amsgrad,
                maximize=self.maximize,
                beta1=self.defaults["beta1"],
                beta2=self.defaults["beta2"],
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                eps=self.defaults["eps"],
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
                has_complex=has_complex,
            )

```



## High-Level Overview


This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_FunctionalAdamW`

**Functions defined**: `__init__`, `step_param`, `step`

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
- [`functional_adagrad.py_docs.md`](./functional_adagrad.py_docs.md)
- [`functional_rprop.py_docs.md`](./functional_rprop.py_docs.md)
- [`functional_adam.py_docs.md`](./functional_adam.py_docs.md)


## Cross-References

- **File Documentation**: `functional_adamw.py_docs.md`
- **Keyword Index**: `functional_adamw.py_kw.md`
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
- [`functional_adagrad.py_docs.md_docs.md`](./functional_adagrad.py_docs.md_docs.md)
- [`zero_redundancy_optimizer.py_docs.md_docs.md`](./zero_redundancy_optimizer.py_docs.md_docs.md)
- [`_deprecation_warning.py_kw.md_docs.md`](./_deprecation_warning.py_kw.md_docs.md)
- [`zero_redundancy_optimizer.py_kw.md_docs.md`](./zero_redundancy_optimizer.py_kw.md_docs.md)
- [`functional_rmsprop.py_docs.md_docs.md`](./functional_rmsprop.py_docs.md_docs.md)
- [`functional_rprop.py_docs.md_docs.md`](./functional_rprop.py_docs.md_docs.md)
- [`named_optimizer.py_docs.md_docs.md`](./named_optimizer.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `functional_adamw.py_docs.md_docs.md`
- **Keyword Index**: `functional_adamw.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
