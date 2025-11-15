# Documentation: `docs/torch/distributions/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributions/__init__.py_docs.md`
- **Size**: 11,763 bytes (11.49 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/distributions/__init__.py`

## File Metadata

- **Path**: `torch/distributions/__init__.py`
- **Size**: 6,111 bytes (5.97 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
r"""
The ``distributions`` package contains parameterizable probability distributions
and sampling functions. This allows the construction of stochastic computation
graphs and stochastic gradient estimators for optimization. This package
generally follows the design of the `TensorFlow Distributions`_ package.

.. _`TensorFlow Distributions`:
    https://arxiv.org/abs/1711.10604

It is not possible to directly backpropagate through random samples. However,
there are two main methods for creating surrogate functions that can be
backpropagated through. These are the score function estimator/likelihood ratio
estimator/REINFORCE and the pathwise derivative estimator. REINFORCE is commonly
seen as the basis for policy gradient methods in reinforcement learning, and the
pathwise derivative estimator is commonly seen in the reparameterization trick
in variational autoencoders. Whilst the score function only requires the value
of samples :math:`f(x)`, the pathwise derivative requires the derivative
:math:`f'(x)`. The next sections discuss these two in a reinforcement learning
example. For more details see
`Gradient Estimation Using Stochastic Computation Graphs`_ .

.. _`Gradient Estimation Using Stochastic Computation Graphs`:
     https://arxiv.org/abs/1506.05254

Score function
^^^^^^^^^^^^^^

When the probability density function is differentiable with respect to its
parameters, we only need :meth:`~torch.distributions.Distribution.sample` and
:meth:`~torch.distributions.Distribution.log_prob` to implement REINFORCE:

.. math::

    \Delta\theta  = \alpha r \frac{\partial\log p(a|\pi^\theta(s))}{\partial\theta}

where :math:`\theta` are the parameters, :math:`\alpha` is the learning rate,
:math:`r` is the reward and :math:`p(a|\pi^\theta(s))` is the probability of
taking action :math:`a` in state :math:`s` given policy :math:`\pi^\theta`.

In practice we would sample an action from the output of a network, apply this
action in an environment, and then use ``log_prob`` to construct an equivalent
loss function. Note that we use a negative because optimizers use gradient
descent, whilst the rule above assumes gradient ascent. With a categorical
policy, the code for implementing REINFORCE would be as follows::

    probs = policy_network(state)
    # Note that this is equivalent to what used to be called multinomial
    m = Categorical(probs)
    action = m.sample()
    next_state, reward = env.step(action)
    loss = -m.log_prob(action) * reward
    loss.backward()

Pathwise derivative
^^^^^^^^^^^^^^^^^^^

The other way to implement these stochastic/policy gradients would be to use the
reparameterization trick from the
:meth:`~torch.distributions.Distribution.rsample` method, where the
parameterized random variable can be constructed via a parameterized
deterministic function of a parameter-free random variable. The reparameterized
sample therefore becomes differentiable. The code for implementing the pathwise
derivative would be as follows::

    params = policy_network(state)
    m = Normal(*params)
    # Any distribution with .has_rsample == True could work based on the application
    action = m.rsample()
    next_state, reward = env.step(action)  # Assuming that reward is differentiable
    loss = -reward
    loss.backward()
"""

from . import transforms
from .bernoulli import Bernoulli
from .beta import Beta
from .binomial import Binomial
from .categorical import Categorical
from .cauchy import Cauchy
from .chi2 import Chi2
from .constraint_registry import biject_to, transform_to
from .continuous_bernoulli import ContinuousBernoulli
from .dirichlet import Dirichlet
from .distribution import Distribution
from .exp_family import ExponentialFamily
from .exponential import Exponential
from .fishersnedecor import FisherSnedecor
from .gamma import Gamma
from .generalized_pareto import GeneralizedPareto
from .geometric import Geometric
from .gumbel import Gumbel
from .half_cauchy import HalfCauchy
from .half_normal import HalfNormal
from .independent import Independent
from .inverse_gamma import InverseGamma
from .kl import _add_kl_info, kl_divergence, register_kl
from .kumaraswamy import Kumaraswamy
from .laplace import Laplace
from .lkj_cholesky import LKJCholesky
from .log_normal import LogNormal
from .logistic_normal import LogisticNormal
from .lowrank_multivariate_normal import LowRankMultivariateNormal
from .mixture_same_family import MixtureSameFamily
from .multinomial import Multinomial
from .multivariate_normal import MultivariateNormal
from .negative_binomial import NegativeBinomial
from .normal import Normal
from .one_hot_categorical import OneHotCategorical, OneHotCategoricalStraightThrough
from .pareto import Pareto
from .poisson import Poisson
from .relaxed_bernoulli import RelaxedBernoulli
from .relaxed_categorical import RelaxedOneHotCategorical
from .studentT import StudentT
from .transformed_distribution import TransformedDistribution
from .transforms import *  # noqa: F403
from .uniform import Uniform
from .von_mises import VonMises
from .weibull import Weibull
from .wishart import Wishart


_add_kl_info()
del _add_kl_info

__all__ = [
    "Bernoulli",
    "Beta",
    "Binomial",
    "Categorical",
    "Cauchy",
    "Chi2",
    "ContinuousBernoulli",
    "Dirichlet",
    "Distribution",
    "Exponential",
    "ExponentialFamily",
    "FisherSnedecor",
    "Gamma",
    "GeneralizedPareto",
    "Geometric",
    "Gumbel",
    "HalfCauchy",
    "HalfNormal",
    "Independent",
    "InverseGamma",
    "Kumaraswamy",
    "LKJCholesky",
    "Laplace",
    "LogNormal",
    "LogisticNormal",
    "LowRankMultivariateNormal",
    "MixtureSameFamily",
    "Multinomial",
    "MultivariateNormal",
    "NegativeBinomial",
    "Normal",
    "OneHotCategorical",
    "OneHotCategoricalStraightThrough",
    "Pareto",
    "RelaxedBernoulli",
    "RelaxedOneHotCategorical",
    "StudentT",
    "Poisson",
    "Uniform",
    "VonMises",
    "Weibull",
    "Wishart",
    "TransformedDistribution",
    "biject_to",
    "kl_divergence",
    "register_kl",
    "transform_to",
]
__all__.extend(transforms.__all__)

```



## High-Level Overview

r"""The ``distributions`` package contains parameterizable probability distributionsand sampling functions. This allows the construction of stochastic computationgraphs and stochastic gradient estimators for optimization. This packagegenerally follows the design of the `TensorFlow Distributions`_ package... _`TensorFlow Distributions`:    https://arxiv.org/abs/1711.10604It is not possible to directly backpropagate through random samples. However,there are two main methods for creating surrogate functions that can bebackpropagated through. These are the score function estimator/likelihood ratioestimator/REINFORCE and the pathwise derivative estimator. REINFORCE is commonlyseen as the basis for policy gradient methods in reinforcement learning, and thepathwise derivative estimator is commonly seen in the reparameterization trickin variational autoencoders. Whilst the score function only requires the valueof samples :math:`f(x)`, the pathwise derivative requires the derivative:math:`f'(x)`. The next sections discuss these two in a reinforcement learningexample. For more details see`Gradient Estimation Using Stochastic Computation Graphs`_ ... _`Gradient Estimation Using Stochastic Computation Graphs`:     https://arxiv.org/abs/1506.05254Score function^^^^^^^^^^^^^^When the probability density function is differentiable with respect to itsparameters, we only need :meth:`~torch.distributions.Distribution.sample` and:meth:`~torch.distributions.Distribution.log_prob` to implement REINFORCE:.. math::    \Delta\theta  = \alpha r \frac{\partial\log p(a|\pi^\theta(s))}{\partial\theta}where :math:`\theta` are the parameters, :math:`\alpha` is the learning rate,:math:`r` is the reward and :math:`p(a|\pi^\theta(s))` is the probability oftaking action :math:`a` in state :math:`s` given policy :math:`\pi^\theta`.In practice we would sample an action from the output of a network, apply thisaction in an environment, and then use ``log_prob`` to construct an equivalentloss function. Note that we use a negative because optimizers use gradientdescent, whilst the rule above assumes gradient ascent. With a categoricalpolicy, the code for implementing REINFORCE would be as follows::    probs = policy_network(state)    # Note that this is equivalent to what used to be called multinomial    m = Categorical(probs)    action = m.sample()    next_state, reward = env.step(action)

This Python file contains 0 class(es) and 0 function(s).

## Detailed Analysis

### Code Structure

**Key imports**: transforms, Bernoulli, Beta, Binomial, Categorical, Cauchy, Chi2, biject_to, transform_to, ContinuousBernoulli, Dirichlet


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `.`: transforms
- `.bernoulli`: Bernoulli
- `.beta`: Beta
- `.binomial`: Binomial
- `.categorical`: Categorical
- `.cauchy`: Cauchy
- `.chi2`: Chi2
- `.constraint_registry`: biject_to, transform_to
- `.continuous_bernoulli`: ContinuousBernoulli
- `.dirichlet`: Dirichlet
- `.distribution`: Distribution
- `.exp_family`: ExponentialFamily
- `.exponential`: Exponential
- `.fishersnedecor`: FisherSnedecor
- `.gamma`: Gamma
- `.generalized_pareto`: GeneralizedPareto
- `.geometric`: Geometric
- `.gumbel`: Gumbel
- `.half_cauchy`: HalfCauchy
- `.half_normal`: HalfNormal
- `.independent`: Independent
- `.inverse_gamma`: InverseGamma
- `.kl`: _add_kl_info, kl_divergence, register_kl
- `.kumaraswamy`: Kumaraswamy
- `.laplace`: Laplace
- `.lkj_cholesky`: LKJCholesky
- `.log_normal`: LogNormal
- `.logistic_normal`: LogisticNormal
- `.lowrank_multivariate_normal`: LowRankMultivariateNormal
- `.mixture_same_family`: MixtureSameFamily


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

Files in the same folder (`torch/distributions`):

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

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributions`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributions`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/distributions`):

- [`wishart.py_docs.md_docs.md`](./wishart.py_docs.md_docs.md)
- [`pareto.py_docs.md_docs.md`](./pareto.py_docs.md_docs.md)
- [`binomial.py_docs.md_docs.md`](./binomial.py_docs.md_docs.md)
- [`half_cauchy.py_docs.md_docs.md`](./half_cauchy.py_docs.md_docs.md)
- [`one_hot_categorical.py_docs.md_docs.md`](./one_hot_categorical.py_docs.md_docs.md)
- [`geometric.py_kw.md_docs.md`](./geometric.py_kw.md_docs.md)
- [`kumaraswamy.py_kw.md_docs.md`](./kumaraswamy.py_kw.md_docs.md)
- [`transformed_distribution.py_kw.md_docs.md`](./transformed_distribution.py_kw.md_docs.md)
- [`log_normal.py_docs.md_docs.md`](./log_normal.py_docs.md_docs.md)
- [`kumaraswamy.py_docs.md_docs.md`](./kumaraswamy.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
