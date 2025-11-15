# Documentation: `docs/torch/optim/_functional.py_docs.md`

## File Metadata

- **Path**: `docs/torch/optim/_functional.py_docs.md`
- **Size**: 6,898 bytes (6.74 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/optim/_functional.py`

## File Metadata

- **Path**: `torch/optim/_functional.py`
- **Size**: 3,250 bytes (3.17 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
r"""Functional interface."""

import math

from torch import Tensor

from .adadelta import adadelta  # type: ignore[attr-defined]  # noqa: F401
from .adagrad import _make_sparse, adagrad  # type: ignore[attr-defined]  # noqa: F401
from .adam import adam  # type: ignore[attr-defined]  # noqa: F401
from .adamax import adamax  # type: ignore[attr-defined]  # noqa: F401
from .adamw import adamw  # type: ignore[attr-defined]  # noqa: F401
from .asgd import asgd  # type: ignore[attr-defined]  # noqa: F401
from .nadam import nadam  # type: ignore[attr-defined]  # noqa: F401
from .radam import radam  # type: ignore[attr-defined]  # noqa: F401
from .rmsprop import rmsprop  # type: ignore[attr-defined]  # noqa: F401
from .rprop import rprop  # type: ignore[attr-defined]  # noqa: F401
from .sgd import sgd  # type: ignore[attr-defined]  # noqa: F401


# TODO: use foreach API in optim._functional to do all the computation


def sparse_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[int],
    *,
    eps: float,
    beta1: float,
    beta2: float,
    lr: float,
    maximize: bool,
) -> None:
    r"""Functional API that performs Sparse Adam algorithm computation.

    See :class:`~torch.optim.SparseAdam` for details.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad
        grad = grad.coalesce()  # the update is non-linear so indices must be unique
        grad_indices = grad._indices()
        grad_values = grad._values()
        if grad_values.numel() == 0:
            # Skip update for empty grad
            continue
        size = grad.size()

        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        def make_sparse(values):
            constructor = grad.new
            if grad_indices.dim() == 0 or values.dim() == 0:
                return constructor().resize_as_(grad)
            return constructor(grad_indices, values, size)

        # Decay the first and second moment running average coefficient
        #      old <- b * old + (1 - b) * new
        # <==> old += (1 - b) * (new - old)
        old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
        exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
        exp_avg.add_(make_sparse(exp_avg_update_values))
        old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
        exp_avg_sq_update_values = (
            grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
        )
        exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

        # Dense addition again is intended, avoiding another sparse_mask
        numer = exp_avg_update_values.add_(old_exp_avg_values)
        exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
        denom = exp_avg_sq_update_values.sqrt_().add_(eps)
        del exp_avg_update_values, exp_avg_sq_update_values

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1

        param.add_(make_sparse(-step_size * numer.div_(denom)))

```



## High-Level Overview

r"""Functional interface."""import mathfrom torch import Tensorfrom .adadelta import adadelta  # type: ignore[attr-defined]  # noqa: F401from .adagrad import _make_sparse, adagrad  # type: ignore[attr-defined]  # noqa: F401from .adam import adam  # type: ignore[attr-defined]  # noqa: F401from .adamax import adamax  # type: ignore[attr-defined]  # noqa: F401from .adamw import adamw  # type: ignore[attr-defined]  # noqa: F401from .asgd import asgd  # type: ignore[attr-defined]  # noqa: F401from .nadam import nadam  # type: ignore[attr-defined]  # noqa: F401from .radam import radam  # type: ignore[attr-defined]  # noqa: F401from .rmsprop import rmsprop  # type: ignore[attr-defined]  # noqa: F401from .rprop import rprop  # type: ignore[attr-defined]  # noqa: F401from .sgd import sgd  # type: ignore[attr-defined]  # noqa: F401# TODO: use foreach API in optim._functional to do all the computationdef sparse_adam(    params: list[Tensor],    grads: list[Tensor],    exp_avgs: list[Tensor],    exp_avg_sqs: list[Tensor],    state_steps: list[int],    *,    eps: float,    beta1: float,    beta2: float,    lr: float,    maximize: bool,) -> None:

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `sparse_adam`, `make_sparse`

**Key imports**: math, Tensor, adadelta  , _make_sparse, adagrad  , adam  , adamax  , adamw  , asgd  , nadam  , radam  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/optim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `math`
- `torch`: Tensor
- `.adadelta`: adadelta  
- `.adagrad`: _make_sparse, adagrad  
- `.adam`: adam  
- `.adamax`: adamax  
- `.adamw`: adamw  
- `.asgd`: asgd  
- `.nadam`: nadam  
- `.radam`: radam  
- `.rmsprop`: rmsprop  
- `.rprop`: rprop  
- `.sgd`: sgd  


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/optim`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`adadelta.py_docs.md`](./adadelta.py_docs.md)
- [`sparse_adam.py_docs.md`](./sparse_adam.py_docs.md)
- [`sgd.py_docs.md`](./sgd.py_docs.md)
- [`lbfgs.py_docs.md`](./lbfgs.py_docs.md)
- [`rmsprop.py_docs.md`](./rmsprop.py_docs.md)
- [`asgd.py_docs.md`](./asgd.py_docs.md)
- [`adagrad.py_docs.md`](./adagrad.py_docs.md)
- [`adam.py_docs.md`](./adam.py_docs.md)


## Cross-References

- **File Documentation**: `_functional.py_docs.md`
- **Keyword Index**: `_functional.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/optim`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/optim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torch/optim`):

- [`rprop.py_kw.md_docs.md`](./rprop.py_kw.md_docs.md)
- [`_muon.py_docs.md_docs.md`](./_muon.py_docs.md_docs.md)
- [`radam.py_kw.md_docs.md`](./radam.py_kw.md_docs.md)
- [`adamw.py_kw.md_docs.md`](./adamw.py_kw.md_docs.md)
- [`adagrad.py_kw.md_docs.md`](./adagrad.py_kw.md_docs.md)
- [`adadelta.py_docs.md_docs.md`](./adadelta.py_docs.md_docs.md)
- [`lbfgs.py_docs.md_docs.md`](./lbfgs.py_docs.md_docs.md)
- [`rmsprop.py_kw.md_docs.md`](./rmsprop.py_kw.md_docs.md)
- [`lbfgs.py_kw.md_docs.md`](./lbfgs.py_kw.md_docs.md)
- [`adamw.py_docs.md_docs.md`](./adamw.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_functional.py_docs.md_docs.md`
- **Keyword Index**: `_functional.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
