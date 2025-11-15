# Documentation: `torch/distributed/optim/functional_rmsprop.py`

## File Metadata

- **Path**: `torch/distributed/optim/functional_rmsprop.py`
- **Size**: 4,684 bytes (4.57 KB)
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


# Define a TorchScript compatible Functional RMSprop Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
@torch.jit.script
class _FunctionalRMSprop:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        foreach: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        self.defaults = {
            "lr": lr,
            "alpha": alpha,
            "eps": eps,
            "weight_decay": weight_decay,
            "momentum": momentum,
        }
        self.centered = centered
        self.foreach = foreach
        self.maximize = maximize

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

        self.state = torch.jit.annotate(dict[torch.Tensor, dict[str, torch.Tensor]], {})

    def step(self, gradients: list[Optional[Tensor]]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        square_avgs = []
        grad_avgs = []
        momentum_buffer_list = []
        state_steps = []
        lr = self.defaults["lr"]
        alpha = self.defaults["alpha"]
        eps = self.defaults["eps"]
        momentum = self.defaults["momentum"]
        weight_decay = self.defaults["weight_decay"]

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        has_complex = False
        for param, gradient in zip(params, gradients):
            if gradient is not None:
                has_complex |= torch.is_complex(param)
                params_with_grad.append(param)
                grads.append(gradient)
                # Lazy state initialization
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state["step"] = torch.tensor(0.0)
                    state["square_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    if momentum > 0:
                        state["momentum_buffer"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )
                    if self.centered:
                        state["grad_avg"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )

                state = self.state[param]
                square_avgs.append(state["square_avg"])
                if momentum > 0:
                    momentum_buffer_list.append(state["momentum_buffer"])
                if self.centered:
                    grad_avgs.append(state["grad_avg"])

                state_steps.append(state["step"])

        with torch.no_grad():
            F.rmsprop(
                params_with_grad,
                grads,
                square_avgs,
                grad_avgs,
                momentum_buffer_list,
                state_steps,
                lr=lr,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                momentum=momentum,
                centered=self.centered,
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
            )

```



## High-Level Overview


This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_FunctionalRMSprop`

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
- [`functional_adagrad.py_docs.md`](./functional_adagrad.py_docs.md)
- [`functional_rprop.py_docs.md`](./functional_rprop.py_docs.md)
- [`functional_adam.py_docs.md`](./functional_adam.py_docs.md)


## Cross-References

- **File Documentation**: `functional_rmsprop.py_docs.md`
- **Keyword Index**: `functional_rmsprop.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
