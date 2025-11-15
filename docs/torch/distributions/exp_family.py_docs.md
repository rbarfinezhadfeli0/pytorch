# Documentation: `torch/distributions/exp_family.py`

## File Metadata

- **Path**: `torch/distributions/exp_family.py`
- **Size**: 2,485 bytes (2.43 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Union

import torch
from torch import Tensor
from torch.distributions.distribution import Distribution


__all__ = ["ExponentialFamily"]


class ExponentialFamily(Distribution):
    r"""
    ExponentialFamily is the abstract base class for probability distributions belonging to an
    exponential family, whose probability mass/density function has the form is defined below

    .. math::

        p_{F}(x; \theta) = \exp(\langle t(x), \theta\rangle - F(\theta) + k(x))

    where :math:`\theta` denotes the natural parameters, :math:`t(x)` denotes the sufficient statistic,
    :math:`F(\theta)` is the log normalizer function for a given family and :math:`k(x)` is the carrier
    measure.

    Note:
        This class is an intermediary between the `Distribution` class and distributions which belong
        to an exponential family mainly to check the correctness of the `.entropy()` and analytic KL
        divergence methods. We use this class to compute the entropy and KL divergence using the AD
        framework and Bregman divergences (courtesy of: Frank Nielsen and Richard Nock, Entropies and
        Cross-entropies of Exponential Families).
    """

    @property
    def _natural_params(self) -> tuple[Tensor, ...]:
        """
        Abstract method for natural parameters. Returns a tuple of Tensors based
        on the distribution
        """
        raise NotImplementedError

    def _log_normalizer(self, *natural_params):
        """
        Abstract method for log normalizer function. Returns a log normalizer based on
        the distribution and input
        """
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self) -> float:
        """
        Abstract method for expected carrier measure, which is required for computing
        entropy.
        """
        raise NotImplementedError

    def entropy(self):
        """
        Method to compute the entropy using Bregman divergence of the log normalizer.
        """
        result: Union[Tensor, float] = -self._mean_carrier_measure
        nparams = [p.detach().requires_grad_() for p in self._natural_params]
        lg_normal = self._log_normalizer(*nparams)
        gradients = torch.autograd.grad(lg_normal.sum(), nparams, create_graph=True)
        result += lg_normal
        for np, g in zip(nparams, gradients):
            result -= (np * g).reshape(self._batch_shape + (-1,)).sum(-1)
        return result

```



## High-Level Overview

r"""    ExponentialFamily is the abstract base class for probability distributions belonging to an    exponential family, whose probability mass/density function has the form is defined below    .. math::        p_{F}(x; \theta) = \exp(\langle t(x), \theta\rangle - F(\theta) + k(x))    where :math:`\theta` denotes the natural parameters, :math:`t(x)` denotes the sufficient statistic,    :math:`F(\theta)` is the log normalizer function for a given family and :math:`k(x)` is the carrier    measure.    Note:        This class is an intermediary between the `Distribution` class and distributions which belong        to an exponential family mainly to check the correctness of the `.entropy()` and analytic KL        divergence methods. We use this class to compute the entropy and KL divergence using the AD        framework and Bregman divergences (courtesy of: Frank Nielsen and Richard Nock, Entropies and        Cross-entropies of Exponential Families).

This Python file contains 5 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ExponentialFamily`

**Functions defined**: `_natural_params`, `_log_normalizer`, `_mean_carrier_measure`, `entropy`

**Key imports**: Union, torch, Tensor, Distribution


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Union
- `torch`
- `torch.distributions.distribution`: Distribution


## Code Patterns & Idioms

### Common Patterns

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

Files in the same folder (`torch/distributions`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mixture_same_family.py_docs.md`](./mixture_same_family.py_docs.md)
- [`normal.py_docs.md`](./normal.py_docs.md)
- [`relaxed_categorical.py_docs.md`](./relaxed_categorical.py_docs.md)
- [`laplace.py_docs.md`](./laplace.py_docs.md)
- [`bernoulli.py_docs.md`](./bernoulli.py_docs.md)
- [`distribution.py_docs.md`](./distribution.py_docs.md)
- [`negative_binomial.py_docs.md`](./negative_binomial.py_docs.md)
- [`continuous_bernoulli.py_docs.md`](./continuous_bernoulli.py_docs.md)
- [`half_normal.py_docs.md`](./half_normal.py_docs.md)


## Cross-References

- **File Documentation**: `exp_family.py_docs.md`
- **Keyword Index**: `exp_family.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
