# Documentation: `torch/distributed/optim/functional_sgd.py`

## File Metadata

- **Path**: `torch/distributed/optim/functional_sgd.py`
- **Size**: 5,938 bytes (5.80 KB)
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


# Define a TorchScript compatible Functional SGD Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
@torch.jit.script
class _FunctionalSGD:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
        foreach: bool = False,
        fused: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        self.defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
        }
        self.nesterov = nesterov
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
        """Similar to self.step, but operates on a single parameter and
        its gradient.
        """
        # TODO: Once step_param interface is robust, refactor step to call
        # step param on each param.
        weight_decay = self.defaults["weight_decay"]
        momentum = self.defaults["momentum"]
        dampening = self.defaults["dampening"]
        lr = self.defaults["lr"]
        params = [param]
        momentum_buffer_list: list[Optional[Tensor]] = []
        grads = []

        has_sparse_grad = False
        if grad is not None:
            grads.append(grad)
            if grad.is_sparse:
                has_sparse_grad = True
            if param not in self.state:
                self.state[param] = {}
            state = self.state[param]
            if "momentum_buffer" not in state:
                momentum_buffer_list.append(None)
            else:
                momentum_buffer_list.append(state["momentum_buffer"])

        with torch.no_grad():
            F.sgd(
                params,
                grads,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=self.nesterov,
                maximize=self.maximize,
                has_sparse_grad=has_sparse_grad,
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
            )
        # update momentum_buffer in state
        state = self.state[param]
        momentum_buffer = momentum_buffer_list[0]
        if momentum_buffer is not None:
            state["momentum_buffer"] = momentum_buffer

    def step(self, gradients: list[Optional[Tensor]]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        momentum_buffer_list: list[Optional[Tensor]] = []
        lr = self.defaults["lr"]
        weight_decay = self.defaults["weight_decay"]
        momentum = self.defaults["momentum"]
        dampening = self.defaults["dampening"]

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        has_sparse_grad = False
        for param, gradient in zip(params, gradients):
            if gradient is not None:
                params_with_grad.append(param)
                grads.append(gradient)
                if gradient.is_sparse:
                    has_sparse_grad = True

                if param not in self.state:
                    self.state[param] = {}

                state = self.state[param]
                if "momentum_buffer" not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state["momentum_buffer"])

        with torch.no_grad():
            F.sgd(
                params_with_grad,
                grads,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=self.nesterov,
                maximize=self.maximize,
                has_sparse_grad=has_sparse_grad,
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
            )

        # update momentum_buffers in state
        for i, p in enumerate(params_with_grad):
            state = self.state[p]
            momentum_buffer = momentum_buffer_list[i]
            if momentum_buffer is not None:
                state["momentum_buffer"] = momentum_buffer

```



## High-Level Overview


This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_FunctionalSGD`

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

- **File Documentation**: `functional_sgd.py_docs.md`
- **Keyword Index**: `functional_sgd.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
