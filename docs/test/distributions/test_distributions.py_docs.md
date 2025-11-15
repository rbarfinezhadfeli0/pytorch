# Documentation: `test/distributions/test_distributions.py`

## File Metadata

- **Path**: `test/distributions/test_distributions.py`
- **Size**: 294,438 bytes (287.54 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: distributions"]

"""
Note [Randomized statistical tests]
-----------------------------------

This note describes how to maintain tests in this file as random sources
change. This file contains two types of randomized tests:

1. The easier type of randomized test are tests that should always pass but are
   initialized with random data. If these fail something is wrong, but it's
   fine to use a fixed seed by inheriting from common.TestCase.

2. The trickier tests are statistical tests. These tests explicitly call
   set_rng_seed(n) and are marked "see Note [Randomized statistical tests]".
   These statistical tests have a known positive failure rate
   (we set failure_rate=1e-3 by default). We need to balance strength of these
   tests with annoyance of false alarms. One way that works is to specifically
   set seeds in each of the randomized tests. When a random generator
   occasionally changes (as in #4312 vectorizing the Box-Muller sampler), some
   of these statistical tests may (rarely) fail. If one fails in this case,
   it's fine to increment the seed of the failing test (but you shouldn't need
   to increment it more than once; otherwise something is probably actually
   wrong).

3. `test_geometric_sample`, `test_binomial_sample` and `test_poisson_sample`
   are validated against `scipy.stats.` which are not guaranteed to be identical
   across different versions of scipy (namely, they yield invalid results in 1.7+)
"""

import math
import numbers
import unittest
from collections import namedtuple
from itertools import product
from random import shuffle

from packaging import version

import torch
import torch.autograd.forward_ad as fwAD
from torch import inf, nan
from torch.autograd import grad
from torch.autograd.functional import jacobian
from torch.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Cauchy,
    Chi2,
    constraints,
    ContinuousBernoulli,
    Dirichlet,
    Distribution,
    Exponential,
    ExponentialFamily,
    FisherSnedecor,
    Gamma,
    GeneralizedPareto,
    Geometric,
    Gumbel,
    HalfCauchy,
    HalfNormal,
    Independent,
    InverseGamma,
    kl_divergence,
    Kumaraswamy,
    Laplace,
    LKJCholesky,
    LogisticNormal,
    LogNormal,
    LowRankMultivariateNormal,
    MixtureSameFamily,
    Multinomial,
    MultivariateNormal,
    NegativeBinomial,
    Normal,
    OneHotCategorical,
    OneHotCategoricalStraightThrough,
    Pareto,
    Poisson,
    RelaxedBernoulli,
    RelaxedOneHotCategorical,
    StudentT,
    TransformedDistribution,
    Uniform,
    VonMises,
    Weibull,
    Wishart,
)
from torch.distributions.constraint_registry import transform_to
from torch.distributions.constraints import Constraint, is_dependent
from torch.distributions.dirichlet import _Dirichlet_backward
from torch.distributions.kl import _kl_expfamily_expfamily
from torch.distributions.transforms import (
    AffineTransform,
    CatTransform,
    ExpTransform,
    identity_transform,
    StackTransform,
)
from torch.distributions.utils import (
    lazy_property,
    probs_to_logits,
    tril_matrix_to_vec,
    vec_to_tril_matrix,
)
from torch.nn.functional import softmax
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    gradcheck,
    load_tests,
    run_tests,
    set_default_dtype,
    set_rng_seed,
    skipIfTorchDynamo,
    TEST_XPU,
    TestCase,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

TEST_NUMPY = True
try:
    import numpy as np
    import scipy.special
    import scipy.stats
except ImportError:
    TEST_NUMPY = False


def pairwise(Dist, *params):
    """
    Creates a pair of distributions `Dist` initialized to test each element of
    param with each other.
    """
    params1 = [torch.tensor([p] * len(p)) for p in params]
    params2 = [p.transpose(0, 1) for p in params1]
    return Dist(*params1), Dist(*params2)


def is_all_nan(tensor):
    """
    Checks if all entries of a tensor is nan.
    """
    return (tensor != tensor).all()


Example = namedtuple("Example", ["Dist", "params"])


# Register all distributions for generic tests by appending to this list.
def _get_examples():
    return [
        Example(
            Bernoulli,
            [
                {"probs": torch.tensor([0.7, 0.2, 0.4], requires_grad=True)},
                {"probs": torch.tensor([0.3], requires_grad=True)},
                {"probs": 0.3},
                {"logits": torch.tensor([0.0], requires_grad=True)},
            ],
        ),
        Example(
            Geometric,
            [
                {"probs": torch.tensor([0.7, 0.2, 0.4], requires_grad=True)},
                {"probs": torch.tensor([0.3], requires_grad=True)},
                {"probs": 0.3},
            ],
        ),
        Example(
            Beta,
            [
                {
                    "concentration1": torch.randn(2, 3).exp().requires_grad_(),
                    "concentration0": torch.randn(2, 3).exp().requires_grad_(),
                },
                {
                    "concentration1": torch.randn(4).exp().requires_grad_(),
                    "concentration0": torch.randn(4).exp().requires_grad_(),
                },
            ],
        ),
        Example(
            Categorical,
            [
                {
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    )
                },
                {"probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)},
                {"logits": torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
            ],
        ),
        Example(
            Binomial,
            [
                {
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    ),
                    "total_count": 10,
                },
                {
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
                    "total_count": 10,
                },
                {
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
                    "total_count": torch.tensor([10]),
                },
                {
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
                    "total_count": torch.tensor([10, 8]),
                },
                {
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
                    "total_count": torch.tensor([[10.0, 8.0], [5.0, 3.0]]),
                },
                {
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
                    "total_count": torch.tensor(0.0),
                },
            ],
        ),
        Example(
            NegativeBinomial,
            [
                {
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    ),
                    "total_count": 10,
                },
                {
                    "probs": torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True),
                    "total_count": 10,
                },
                {
                    "probs": torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True),
                    "total_count": torch.tensor([10]),
                },
                {
                    "probs": torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True),
                    "total_count": torch.tensor([10, 8]),
                },
                {
                    "probs": torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True),
                    "total_count": torch.tensor([[10.0, 8.0], [5.0, 3.0]]),
                },
                {
                    "probs": torch.tensor([[0.9, 0.0], [0.0, 0.9]], requires_grad=True),
                    "total_count": torch.tensor(0.0),
                },
            ],
        ),
        Example(
            Multinomial,
            [
                {
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    ),
                    "total_count": 10,
                },
                {
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
                    "total_count": 10,
                },
            ],
        ),
        Example(
            Cauchy,
            [
                {"loc": 0.0, "scale": 1.0},
                {"loc": torch.tensor([0.0]), "scale": 1.0},
                {
                    "loc": torch.tensor([[0.0], [0.0]]),
                    "scale": torch.tensor([[1.0], [1.0]]),
                },
            ],
        ),
        Example(
            Chi2,
            [
                {"df": torch.randn(2, 3).exp().requires_grad_()},
                {"df": torch.randn(1).exp().requires_grad_()},
            ],
        ),
        Example(
            StudentT,
            [
                {"df": torch.randn(2, 3).exp().requires_grad_()},
                {"df": torch.randn(1).exp().requires_grad_()},
            ],
        ),
        Example(
            Dirichlet,
            [
                {"concentration": torch.randn(2, 3).exp().requires_grad_()},
                {"concentration": torch.randn(4).exp().requires_grad_()},
            ],
        ),
        Example(
            Exponential,
            [
                {"rate": torch.randn(5, 5).abs().requires_grad_()},
                {"rate": torch.randn(1).abs().requires_grad_()},
            ],
        ),
        Example(
            FisherSnedecor,
            [
                {
                    "df1": torch.randn(5, 5).abs().requires_grad_(),
                    "df2": torch.randn(5, 5).abs().requires_grad_(),
                },
                {
                    "df1": torch.randn(1).abs().requires_grad_(),
                    "df2": torch.randn(1).abs().requires_grad_(),
                },
                {
                    "df1": torch.tensor([1.0]),
                    "df2": 1.0,
                },
            ],
        ),
        Example(
            Gamma,
            [
                {
                    "concentration": torch.randn(2, 3).exp().requires_grad_(),
                    "rate": torch.randn(2, 3).exp().requires_grad_(),
                },
                {
                    "concentration": torch.randn(1).exp().requires_grad_(),
                    "rate": torch.randn(1).exp().requires_grad_(),
                },
            ],
        ),
        Example(
            Gumbel,
            [
                {
                    "loc": torch.randn(5, 5, requires_grad=True),
                    "scale": torch.randn(5, 5).abs().requires_grad_(),
                },
                {
                    "loc": torch.randn(1, requires_grad=True),
                    "scale": torch.randn(1).abs().requires_grad_(),
                },
            ],
        ),
        Example(HalfCauchy, [{"scale": 1.0}, {"scale": torch.tensor([[1.0], [1.0]])}]),
        Example(
            HalfNormal,
            [
                {"scale": torch.randn(5, 5).abs().requires_grad_()},
                {"scale": torch.randn(1).abs().requires_grad_()},
                {"scale": torch.tensor([1e-5, 1e-5], requires_grad=True)},
            ],
        ),
        Example(
            Independent,
            [
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, requires_grad=True),
                        torch.randn(2, 3).abs().requires_grad_(),
                    ),
                    "reinterpreted_batch_ndims": 0,
                },
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, requires_grad=True),
                        torch.randn(2, 3).abs().requires_grad_(),
                    ),
                    "reinterpreted_batch_ndims": 1,
                },
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, requires_grad=True),
                        torch.randn(2, 3).abs().requires_grad_(),
                    ),
                    "reinterpreted_batch_ndims": 2,
                },
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, 5, requires_grad=True),
                        torch.randn(2, 3, 5).abs().requires_grad_(),
                    ),
                    "reinterpreted_batch_ndims": 2,
                },
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, 5, requires_grad=True),
                        torch.randn(2, 3, 5).abs().requires_grad_(),
                    ),
                    "reinterpreted_batch_ndims": 3,
                },
            ],
        ),
        Example(
            Kumaraswamy,
            [
                {
                    "concentration1": torch.empty(2, 3).uniform_(1, 2).requires_grad_(),
                    "concentration0": torch.empty(2, 3).uniform_(1, 2).requires_grad_(),
                },
                {
                    "concentration1": torch.rand(4).uniform_(1, 2).requires_grad_(),
                    "concentration0": torch.rand(4).uniform_(1, 2).requires_grad_(),
                },
            ],
        ),
        Example(
            LKJCholesky,
            [
                {"dim": 2, "concentration": 0.5},
                {
                    "dim": 3,
                    "concentration": torch.tensor([0.5, 1.0, 2.0]),
                },
                {"dim": 100, "concentration": 4.0},
            ],
        ),
        Example(
            Laplace,
            [
                {
                    "loc": torch.randn(5, 5, requires_grad=True),
                    "scale": torch.randn(5, 5).abs().requires_grad_(),
                },
                {
                    "loc": torch.randn(1, requires_grad=True),
                    "scale": torch.randn(1).abs().requires_grad_(),
                },
                {
                    "loc": torch.tensor([1.0, 0.0], requires_grad=True),
                    "scale": torch.tensor([1e-5, 1e-5], requires_grad=True),
                },
            ],
        ),
        Example(
            LogNormal,
            [
                {
                    "loc": torch.randn(5, 5, requires_grad=True),
                    "scale": torch.randn(5, 5).abs().requires_grad_(),
                },
                {
                    "loc": torch.randn(1, requires_grad=True),
                    "scale": torch.randn(1).abs().requires_grad_(),
                },
                {
                    "loc": torch.tensor([1.0, 0.0], requires_grad=True),
                    "scale": torch.tensor([1e-5, 1e-5], requires_grad=True),
                },
            ],
        ),
        Example(
            LogisticNormal,
            [
                {
                    "loc": torch.randn(5, 5).requires_grad_(),
                    "scale": torch.randn(5, 5).abs().requires_grad_(),
                },
                {
                    "loc": torch.randn(1).requires_grad_(),
                    "scale": torch.randn(1).abs().requires_grad_(),
                },
                {
                    "loc": torch.tensor([1.0, 0.0], requires_grad=True),
                    "scale": torch.tensor([1e-5, 1e-5], requires_grad=True),
                },
            ],
        ),
        Example(
            LowRankMultivariateNormal,
            [
                {
                    "loc": torch.randn(5, 2, requires_grad=True),
                    "cov_factor": torch.randn(5, 2, 1, requires_grad=True),
                    "cov_diag": torch.tensor([2.0, 0.25], requires_grad=True),
                },
                {
                    "loc": torch.randn(4, 3, requires_grad=True),
                    "cov_factor": torch.randn(3, 2, requires_grad=True),
                    "cov_diag": torch.tensor([5.0, 1.5, 3.0], requires_grad=True),
                },
            ],
        ),
        Example(
            MultivariateNormal,
            [
                {
                    "loc": torch.randn(5, 2, requires_grad=True),
                    "covariance_matrix": torch.tensor(
                        [[2.0, 0.3], [0.3, 0.25]], requires_grad=True
                    ),
                },
                {
                    "loc": torch.randn(2, 3, requires_grad=True),
                    "precision_matrix": torch.tensor(
                        [[2.0, 0.1, 0.0], [0.1, 0.25, 0.0], [0.0, 0.0, 0.3]],
                        requires_grad=True,
                    ),
                },
                {
                    "loc": torch.randn(5, 3, 2, requires_grad=True),
                    "scale_tril": torch.tensor(
                        [
                            [[2.0, 0.0], [-0.5, 0.25]],
                            [[2.0, 0.0], [0.3, 0.25]],
                            [[5.0, 0.0], [-0.5, 1.5]],
                        ],
                        requires_grad=True,
                    ),
                },
                {
                    "loc": torch.tensor([1.0, -1.0]),
                    "covariance_matrix": torch.tensor([[5.0, -0.5], [-0.5, 1.5]]),
                },
            ],
        ),
        Example(
            Normal,
            [
                {
                    "loc": torch.randn(5, 5, requires_grad=True),
                    "scale": torch.randn(5, 5).abs().requires_grad_(),
                },
                {
                    "loc": torch.randn(1, requires_grad=True),
                    "scale": torch.randn(1).abs().requires_grad_(),
                },
                {
                    "loc": torch.tensor([1.0, 0.0], requires_grad=True),
                    "scale": torch.tensor([1e-5, 1e-5], requires_grad=True),
                },
            ],
        ),
        Example(
            OneHotCategorical,
            [
                {
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    )
                },
                {"probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)},
                {"logits": torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
            ],
        ),
        Example(
            OneHotCategoricalStraightThrough,
            [
                {
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    )
                },
                {"probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)},
                {"logits": torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
            ],
        ),
        Example(
            Pareto,
            [
                {"scale": 1.0, "alpha": 1.0},
                {
                    "scale": (torch.randn(5, 5).abs() + 0.1).requires_grad_(),
                    "alpha": (torch.randn(5, 5).abs() + 0.1).requires_grad_(),
                },
                {"scale": torch.tensor([1.0]), "alpha": 1.0},
            ],
        ),
        Example(
            Poisson,
            [
                {
                    "rate": torch.randn(5, 5).abs().requires_grad_(),
                },
                {
                    "rate": torch.randn(3).abs().requires_grad_(),
                },
                {
                    "rate": 0.2,
                },
                {
                    "rate": torch.tensor([0.0], requires_grad=True),
                },
                {
                    "rate": 0.0,
                },
            ],
        ),
        Example(
            RelaxedBernoulli,
            [
                {
                    "temperature": torch.tensor([0.5], requires_grad=True),
                    "probs": torch.tensor([0.7, 0.2, 0.4], requires_grad=True),
                },
                {
                    "temperature": torch.tensor([2.0]),
                    "probs": torch.tensor([0.3]),
                },
                {
                    "temperature": torch.tensor([7.2]),
                    "logits": torch.tensor([-2.0, 2.0, 1.0, 5.0]),
                },
            ],
        ),
        Example(
            RelaxedOneHotCategorical,
            [
                {
                    "temperature": torch.tensor([0.5], requires_grad=True),
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]], requires_grad=True
                    ),
                },
                {
                    "temperature": torch.tensor([2.0]),
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                },
                {
                    "temperature": torch.tensor([7.2]),
                    "logits": torch.tensor([[-2.0, 2.0], [1.0, 5.0]]),
                },
            ],
        ),
        Example(
            TransformedDistribution,
            [
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, requires_grad=True),
                        torch.randn(2, 3).abs().requires_grad_(),
                    ),
                    "transforms": [],
                },
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, requires_grad=True),
                        torch.randn(2, 3).abs().requires_grad_(),
                    ),
                    "transforms": ExpTransform(),
                },
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, 5, requires_grad=True),
                        torch.randn(2, 3, 5).abs().requires_grad_(),
                    ),
                    "transforms": [
                        AffineTransform(torch.randn(3, 5), torch.randn(3, 5)),
                        ExpTransform(),
                    ],
                },
                {
                    "base_distribution": Normal(
                        torch.randn(2, 3, 5, requires_grad=True),
                        torch.randn(2, 3, 5).abs().requires_grad_(),
                    ),
                    "transforms": AffineTransform(1, 2),
                },
                {
                    "base_distribution": Uniform(
                        torch.tensor(1e8).log(), torch.tensor(1e10).log()
                    ),
                    "transforms": ExpTransform(),
                },
            ],
        ),
        Example(
            Uniform,
            [
                {
                    "low": torch.zeros(5, 5, requires_grad=True),
                    "high": torch.ones(5, 5, requires_grad=True),
                },
                {
                    "low": torch.zeros(1, requires_grad=True),
                    "high": torch.ones(1, requires_grad=True),
                },
                {
                    "low": torch.tensor([1.0, 1.0], requires_grad=True),
                    "high": torch.tensor([2.0, 3.0], requires_grad=True),
                },
            ],
        ),
        Example(
            Weibull,
            [
                {
                    "scale": torch.randn(5, 5).abs().requires_grad_(),
                    "concentration": torch.randn(1).abs().requires_grad_(),
                }
            ],
        ),
        Example(
            Wishart,
            [
                {
                    "covariance_matrix": torch.tensor(
                        [[2.0, 0.3], [0.3, 0.25]], requires_grad=True
                    ),
                    "df": torch.tensor([3.0], requires_grad=True),
                },
                {
                    "precision_matrix": torch.tensor(
                        [[2.0, 0.1, 0.0], [0.1, 0.25, 0.0], [0.0, 0.0, 0.3]],
                        requires_grad=True,
                    ),
                    "df": torch.tensor([5.0, 4], requires_grad=True),
                },
                {
                    "scale_tril": torch.tensor(
                        [
                            [[2.0, 0.0], [-0.5, 0.25]],
                            [[2.0, 0.0], [0.3, 0.25]],
                            [[5.0, 0.0], [-0.5, 1.5]],
                        ],
                        requires_grad=True,
                    ),
                    "df": torch.tensor([5.0, 3.5, 3], requires_grad=True),
                },
                {
                    "covariance_matrix": torch.tensor([[5.0, -0.5], [-0.5, 1.5]]),
                    "df": torch.tensor([3.0]),
                },
                {
                    "covariance_matrix": torch.tensor([[5.0, -0.5], [-0.5, 1.5]]),
                    "df": 3.0,
                },
            ],
        ),
        Example(
            MixtureSameFamily,
            [
                {
                    "mixture_distribution": Categorical(
                        torch.rand(5, requires_grad=True)
                    ),
                    "component_distribution": Normal(
                        torch.randn(5, requires_grad=True),
                        torch.rand(5, requires_grad=True),
                    ),
                },
                {
                    "mixture_distribution": Categorical(
                        torch.rand(5, requires_grad=True)
                    ),
                    "component_distribution": MultivariateNormal(
                        loc=torch.randn(5, 2, requires_grad=True),
                        covariance_matrix=torch.tensor(
                            [[2.0, 0.3], [0.3, 0.25]], requires_grad=True
                        ),
                    ),
                },
            ],
        ),
        Example(
            VonMises,
            [
                {
                    "loc": torch.tensor(1.0, requires_grad=True),
                    "concentration": torch.tensor(10.0, requires_grad=True),
                },
                {
                    "loc": torch.tensor([0.0, math.pi / 2], requires_grad=True),
                    "concentration": torch.tensor([1.0, 10.0], requires_grad=True),
                },
            ],
        ),
        Example(
            ContinuousBernoulli,
            [
                {"probs": torch.tensor([0.7, 0.2, 0.4], requires_grad=True)},
                {"probs": torch.tensor([0.3], requires_grad=True)},
                {"probs": 0.3},
                {"logits": torch.tensor([0.0], requires_grad=True)},
            ],
        ),
        Example(
            InverseGamma,
            [
                {
                    "concentration": torch.randn(2, 3).exp().requires_grad_(),
                    "rate": torch.randn(2, 3).exp().requires_grad_(),
                },
                {
                    "concentration": torch.randn(1).exp().requires_grad_(),
                    "rate": torch.randn(1).exp().requires_grad_(),
                },
            ],
        ),
        Example(
            GeneralizedPareto,
            [
                {
                    "loc": torch.randn(5, 5, requires_grad=True).mul(10),
                    "scale": torch.randn(5, 5).abs().requires_grad_(),
                    "concentration": torch.randn(5, 5).div(10).requires_grad_(),
                },
            ],
        ),
    ]


# Register all distributions for bad examples by appending to this list.
def _get_bad_examples():
    return [
        Example(
            Bernoulli,
            [
                {"probs": torch.tensor([1.1, 0.2, 0.4], requires_grad=True)},
                {"probs": torch.tensor([-0.5], requires_grad=True)},
                {"probs": 1.00001},
            ],
        ),
        Example(
            Beta,
            [
                {
                    "concentration1": torch.tensor([0.0], requires_grad=True),
                    "concentration0": torch.tensor([0.0], requires_grad=True),
                },
                {
                    "concentration1": torch.tensor([-1.0], requires_grad=True),
                    "concentration0": torch.tensor([-2.0], requires_grad=True),
                },
            ],
        ),
        Example(
            Geometric,
            [
                {"probs": torch.tensor([1.1, 0.2, 0.4], requires_grad=True)},
                {"probs": torch.tensor([-0.3], requires_grad=True)},
                {"probs": 1.00000001},
            ],
        ),
        Example(
            Categorical,
            [
                {
                    "probs": torch.tensor(
                        [[-0.1, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    )
                },
                {
                    "probs": torch.tensor(
                        [[-1.0, 10.0], [0.0, -1.0]], requires_grad=True
                    )
                },
            ],
        ),
        Example(
            Binomial,
            [
                {
                    "probs": torch.tensor(
                        [[-0.0000001, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    ),
                    "total_count": 10,
                },
                {
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True),
                    "total_count": 10,
                },
            ],
        ),
        Example(
            NegativeBinomial,
            [
                {
                    "probs": torch.tensor(
                        [[-0.0000001, 0.2, 0.3], [0.5, 0.3, 0.2]], requires_grad=True
                    ),
                    "total_count": 10,
                },
                {
                    "probs": torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True),
                    "total_count": 10,
                },
            ],
        ),
        Example(
            Cauchy,
            [
                {"loc": 0.0, "scale": -1.0},
                {"loc": torch.tensor([0.0]), "scale": 0.0},
                {
                    "loc": torch.tensor([[0.0], [-2.0]]),
                    "scale": torch.tensor([[-0.000001], [1.0]]),
                },
            ],
        ),
        Example(
            Chi2,
            [
                {"df": torch.tensor([0.0], requires_grad=True)},
                {"df": torch.tensor([-2.0], requires_grad=True)},
            ],
        ),
        Example(
            StudentT,
            [
                {"df": torch.tensor([0.0], requires_grad=True)},
                {"df": torch.tensor([-2.0], requires_grad=True)},
            ],
        ),
        Example(
            Dirichlet,
            [
                {"concentration": torch.tensor([0.0], requires_grad=True)},
                {"concentration": torch.tensor([-2.0], requires_grad=True)},
            ],
        ),
        Example(
            Exponential,
            [
                {"rate": torch.tensor([0.0, 0.0], requires_grad=True)},
                {"rate": torch.tensor([-2.0], requires_grad=True)},
            ],
        ),
        Example(
            FisherSnedecor,
            [
                {
                    "df1": torch.tensor([0.0, 0.0], requires_grad=True),
                    "df2": torch.tensor([-1.0, -100.0], requires_grad=True),
                },
                {
                    "df1": torch.tensor([1.0, 1.0], requires_grad=True),
                    "df2": torch.tensor([0.0, 0.0], requires_grad=True),
                },
            ],
        ),
        Example(
            Gamma,
            [
                {
                    "concentration": torch.tensor([0.0, 0.0], requires_grad=True),
                    "rate": torch.tensor([-1.0, -100.0], requires_grad=True),
                },
                {
                    "concentration": torch.tensor([1.0, 1.0], requires_grad=True),
                    "rate": torch.tensor([0.0, 0.0], requires_grad=True),
                },
            ],
        ),
        Example(
            Gumbel,
            [
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([0.0, 1.0], requires_grad=True),
                },
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([1.0, -1.0], requires_grad=True),
                },
            ],
        ),
        Example(
            HalfCauchy,
            [
                {"scale": -1.0},
                {"scale": 0.0},
                {"scale": torch.tensor([[-0.000001], [1.0]])},
            ],
        ),
        Example(
            HalfNormal,
            [
                {"scale": torch.tensor([0.0, 1.0], requires_grad=True)},
                {"scale": torch.tensor([1.0, -1.0], requires_grad=True)},
            ],
        ),
        Example(
            LKJCholesky,
            [
                {"dim": -2, "concentration": 0.1},
                {
                    "dim": 1,
                    "concentration": 2.0,
                },
                {
                    "dim": 2,
                    "concentration": 0.0,
                },
            ],
        ),
        Example(
            Laplace,
            [
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([0.0, 1.0], requires_grad=True),
                },
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([1.0, -1.0], requires_grad=True),
                },
            ],
        ),
        Example(
            LogNormal,
            [
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([0.0, 1.0], requires_grad=True),
                },
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([1.0, -1.0], requires_grad=True),
                },
            ],
        ),
        Example(
            MultivariateNormal,
            [
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "covariance_matrix": torch.tensor(
                        [[1.0, 0.0], [0.0, -2.0]], requires_grad=True
                    ),
                },
            ],
        ),
        Example(
            Normal,
            [
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([0.0, 1.0], requires_grad=True),
                },
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([1.0, -1.0], requires_grad=True),
                },
                {
                    "loc": torch.tensor([1.0, 0.0], requires_grad=True),
                    "scale": torch.tensor([1e-5, -1e-5], requires_grad=True),
                },
            ],
        ),
        Example(
            OneHotCategorical,
            [
                {
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.1, -10.0, 0.2]], requires_grad=True
                    )
                },
                {"probs": torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
            ],
        ),
        Example(
            OneHotCategoricalStraightThrough,
            [
                {
                    "probs": torch.tensor(
                        [[0.1, 0.2, 0.3], [0.1, -10.0, 0.2]], requires_grad=True
                    )
                },
                {"probs": torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)},
            ],
        ),
        Example(
            Pareto,
            [
                {"scale": 0.0, "alpha": 0.0},
                {
                    "scale": torch.tensor([0.0, 0.0], requires_grad=True),
                    "alpha": torch.tensor([-1e-5, 0.0], requires_grad=True),
                },
                {"scale": torch.tensor([1.0]), "alpha": -1.0},
            ],
        ),
        Example(
            Poisson,
            [
                {
                    "rate": torch.tensor([-0.1], requires_grad=True),
                },
                {
                    "rate": -1.0,
                },
            ],
        ),
        Example(
            RelaxedBernoulli,
            [
                {
                    "temperature": torch.tensor([1.5], requires_grad=True),
                    "probs": torch.tensor([1.7, 0.2, 0.4], requires_grad=True),
                },
                {
                    "temperature": torch.tensor([2.0]),
                    "probs": torch.tensor([-1.0]),
                },
            ],
        ),
        Example(
            RelaxedOneHotCategorical,
            [
                {
                    "temperature": torch.tensor([0.5], requires_grad=True),
                    "probs": torch.tensor(
                        [[-0.1, 0.2, 0.7], [0.5, 0.3, 0.2]], requires_grad=True
                    ),
                },
                {
                    "temperature": torch.tensor([2.0]),
                    "probs": torch.tensor([[-1.0, 0.0], [-1.0, 1.1]]),
                },
            ],
        ),
        Example(
            TransformedDistribution,
            [
                {
                    "base_distribution": Normal(0, 1),
                    "transforms": lambda x: x,
                },
                {
                    "base_distribution": Normal(0, 1),
                    "transforms": [lambda x: x],
                },
            ],
        ),
        Example(
            Uniform,
            [
                {
                    "low": torch.tensor([2.0], requires_grad=True),
                    "high": torch.tensor([2.0], requires_grad=True),
                },
                {
                    "low": torch.tensor([0.0], requires_grad=True),
                    "high": torch.tensor([0.0], requires_grad=True),
                },
                {
                    "low": torch.tensor([1.0], requires_grad=True),
                    "high": torch.tensor([0.0], requires_grad=True),
                },
            ],
        ),
        Example(
            Weibull,
            [
                {
                    "scale": torch.tensor([0.0], requires_grad=True),
                    "concentration": torch.tensor([0.0], requires_grad=True),
                },
                {
                    "scale": torch.tensor([1.0], requires_grad=True),
                    "concentration": torch.tensor([-1.0], requires_grad=True),
                },
            ],
        ),
        Example(
            Wishart,
            [
                {
                    "covariance_matrix": torch.tensor(
                        [[1.0, 0.0], [0.0, -2.0]], requires_grad=True
                    ),
                    "df": torch.tensor([1.5], requires_grad=True),
                },
                {
                    "covariance_matrix": torch.tensor(
                        [[1.0, 1.0], [1.0, -2.0]], requires_grad=True
                    ),
                    "df": torch.tensor([3.0], requires_grad=True),
                },
                {
                    "covariance_matrix": torch.tensor(
                        [[1.0, 1.0], [1.0, -2.0]], requires_grad=True
                    ),
                    "df": 3.0,
                },
            ],
        ),
        Example(
            ContinuousBernoulli,
            [
                {"probs": torch.tensor([1.1, 0.2, 0.4], requires_grad=True)},
                {"probs": torch.tensor([-0.5], requires_grad=True)},
                {"probs": 1.00001},
            ],
        ),
        Example(
            InverseGamma,
            [
                {
                    "concentration": torch.tensor([0.0, 0.0], requires_grad=True),
                    "rate": torch.tensor([-1.0, -100.0], requires_grad=True),
                },
                {
                    "concentration": torch.tensor([1.0, 1.0], requires_grad=True),
                    "rate": torch.tensor([0.0, 0.0], requires_grad=True),
                },
            ],
        ),
        Example(
            GeneralizedPareto,
            [
                {
                    "loc": torch.tensor([0.0, 0.0], requires_grad=True),
                    "scale": torch.tensor([-1.0, -100.0], requires_grad=True),
                    "concentration": torch.tensor([0.0, 0.0], requires_grad=True),
                },
                {
                    "loc": torch.tensor([1.0, 1.0], requires_grad=True),
                    "scale": torch.tensor([0.0, 0.0], requires_grad=True),
                    "concentration": torch.tensor([-1.0, -100.0], requires_grad=True),
                },
            ],
        ),
    ]


class DistributionsTestCase(TestCase):
    def setUp(self):
        """The tests assume that the validation flag is set."""
        torch.distributions.Distribution.set_default_validate_args(True)
        super().setUp()


@skipIfTorchDynamo("Not a TorchDynamo suitable test")
class TestDistributions(DistributionsTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def _gradcheck_log_prob(self, dist_ctor, ctor_params):
        # performs gradient checks on log_prob
        distribution = dist_ctor(*ctor_params)
        s = distribution.sample()
        if not distribution.support.is_discrete:
            s = s.detach().requires_grad_()

        expected_shape = distribution.batch_shape + distribution.event_shape
        self.assertEqual(s.size(), expected_shape)

        def apply_fn(s, *params):
            return dist_ctor(*params).log_prob(s)

        gradcheck(apply_fn, (s,) + tuple(ctor_params), raise_exception=True)

    def _check_forward_ad(self, fn):
        with fwAD.dual_level():
            x = torch.tensor(1.0)
            t = torch.tensor(1.0)
            dual = fwAD.make_dual(x, t)
            dual_out = fn(dual)
            self.assertEqual(
                torch.count_nonzero(fwAD.unpack_dual(dual_out).tangent).item(), 0
            )

    def _check_log_prob(self, dist, asset_fn):
        # checks that the log_prob matches a reference function
        s = dist.sample()
        log_probs = dist.log_prob(s)
        log_probs_data_flat = log_probs.view(-1)
        s_data_flat = s.view(len(log_probs_data_flat), -1)
        for i, (val, log_prob) in enumerate(zip(s_data_flat, log_probs_data_flat)):
            asset_fn(i, val.squeeze(), log_prob)

    def _check_sampler_sampler(
        self,
        torch_dist,
        ref_dist,
        message,
        multivariate=False,
        circular=False,
        num_samples=10000,
        failure_rate=1e-3,
    ):
        # Checks that the .sample() method matches a reference function.
        torch_samples = torch_dist.sample((num_samples,)).squeeze()
        torch_samples = torch_samples.cpu().numpy()
        ref_samples = ref_dist.rvs(num_samples).astype(np.float64)
        if multivariate:
            # Project onto a random axis.
            axis = np.random.normal(size=(1,) + torch_samples.shape[1:])
            axis /= np.linalg.norm(axis)
            torch_samples = (axis * torch_samples).reshape(num_samples, -1).sum(-1)
            ref_samples = (axis * ref_samples).reshape(num_samples, -1).sum(-1)
        samples = [(x, +1) for x in torch_samples] + [(x, -1) for x in ref_samples]
        if circular:
            samples = [(np.cos(x), v) for (x, v) in samples]
        shuffle(
            samples
        )  # necessary to prevent stable sort from making uneven bins for discrete
        samples.sort(key=lambda x: x[0])
        samples = np.array(samples)[:, 1]

        # Aggregate into bins filled with roughly zero-mean unit-variance RVs.
        num_bins = 10
        samples_per_bin = len(samples) // num_bins
        bins = samples.reshape((num_bins, samples_per_bin)).mean(axis=1)
        stddev = samples_per_bin**-0.5
        threshold = stddev * scipy.special.erfinv(1 - 2 * failure_rate / num_bins)
        message = f"{message}.sample() is biased:\n{bins}"
        for bias in bins:
            self.assertLess(-threshold, bias, message)
            self.assertLess(bias, threshold, message)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def _check_sampler_discrete(
        self, torch_dist, ref_dist, message, num_samples=10000, failure_rate=1e-3
    ):
        """Runs a Chi2-test for the support, but ignores tail instead of combining"""
        torch_samples = torch_dist.sample((num_samples,)).squeeze()
        torch_samples = (
            torch_samples.float()
            if torch_samples.dtype == torch.bfloat16
            else torch_samples
        )
        torch_samples = torch_samples.cpu().numpy()
        unique, counts = np.unique(torch_samples, return_counts=True)
        pmf = ref_dist.pmf(unique)
        pmf = pmf / pmf.sum()  # renormalize to 1.0 for chisq test
        msk = (counts > 5) & ((pmf * num_samples) > 5)
        self.assertGreater(
            pmf[msk].sum(),
            0.9,
            "Distribution is too sparse for test; try increasing num_samples",
        )
        # Add a remainder bucket that combines counts for all values
        # below threshold, if such values exist (i.e. mask has False entries).
        if not msk.all():
            counts = np.concatenate([counts[msk], np.sum(counts[~msk], keepdims=True)])
            pmf = np.concatenate([pmf[msk], np.sum(pmf[~msk], keepdims=True)])
        _, p = scipy.stats.chisquare(counts, pmf * num_samples)
        self.assertGreater(p, failure_rate, message)

    def _check_enumerate_support(self, dist, examples):
        for params, expected in examples:
            params = {k: torch.tensor(v) for k, v in params.items()}
            d = dist(**params)
            actual = d.enumerate_support(expand=False)
            expected = torch.tensor(expected, dtype=actual.dtype)
            self.assertEqual(actual, expected)
            actual = d.enumerate_support(expand=True)
            expected_with_expand = expected.expand(
                (-1,) + d.batch_shape + d.event_shape
            )
            self.assertEqual(actual, expected_with_expand)

    def test_repr(self):
        for Dist, params in _get_examples():
            for param in params:
                dist = Dist(**param)
                self.assertTrue(repr(dist).startswith(dist.__class__.__name__))

    def test_sample_detached(self):
        for Dist, params in _get_examples():
            for i, param in enumerate(params):
                variable_params = [
                    p for p in param.values() if getattr(p, "requires_grad", False)
                ]
                if not variable_params:
                    continue
                dist = Dist(**param)
                sample = dist.sample()
                self.assertFalse(
                    sample.requires_grad,
                    msg=f"{Dist.__name__} example {i + 1}/{len(params)}, .sample() is not detached",
                )

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    def test_rsample_requires_grad(self):
        for Dist, params in _get_examples():
            for i, param in enumerate(params):
                if not any(getattr(p, "requires_grad", False) for p in param.values()):
                    continue
                dist = Dist(**param)
                if not dist.has_rsample:
                    continue
                sample = dist.rsample()
                self.assertTrue(
                    sample.requires_grad,
                    msg=f"{Dist.__name__} example {i + 1}/{len(params)}, .rsample() does not require grad",
                )

    def test_enumerate_support_type(self):
        for Dist, params in _get_examples():
            for i, param in enumerate(params):
                dist = Dist(**param)
                try:
                    self.assertTrue(
                        type(dist.sample()) is type(dist.enumerate_support()),
                        msg=(
                            "{} example {}/{}, return type mismatch between "
                            + "sample and enumerate_support."
                        ).format(Dist.__name__, i + 1, len(params)),
                    )
                except NotImplementedError:
                    pass

    def test_lazy_property_grad(self):
        x = torch.randn(1, requires_grad=True)

        class Dummy:
            @lazy_property
            def y(self):
                return x + 1

        def test():
            x.grad = None
            Dummy().y.backward()
            self.assertEqual(x.grad, torch.ones(1))

        test()
        with torch.no_grad():
            test()

        mean = torch.randn(2)
        cov = torch.eye(2, requires_grad=True)
        distn = MultivariateNormal(mean, cov)
        with torch.no_grad():
            distn.scale_tril
        distn.scale_tril.sum().backward()
        self.assertIsNotNone(cov.grad)

    def test_has_examples(self):
        distributions_with_examples = {e.Dist for e in _get_ex
```



## High-Level Overview

"""Note [Randomized statistical tests]-----------------------------------This note describes how to maintain tests in this file as random sourceschange. This file contains two types of randomized tests:1. The easier type of randomized test are tests that should always pass but are   initialized with random data. If these fail something is wrong, but it's   fine to use a fixed seed by inheriting from common.TestCase.2. The trickier tests are statistical tests. These tests explicitly call   set_rng_seed(n) and are marked "see Note [Randomized statistical tests]".   These statistical tests have a known positive failure rate   (we set failure_rate=1e-3 by default). We need to balance strength of these   tests with annoyance of false alarms. One way that works is to specifically   set seeds in each of the randomized tests. When a random generator   occasionally changes (as in #4312 vectorizing the Box-Muller sampler), some   of these statistical tests may (rarely) fail. If one fails in this case,   it's fine to increment the seed of the failing test (but you shouldn't need   to increment it more than once; otherwise something is probably actually   wrong).3. `test_geometric_sample`, `test_binomial_sample` and `test_poisson_sample`   are validated against `scipy.stats.` which are not guaranteed to be identical   across different versions of scipy (namely, they yield invalid results in 1.7+)

This Python file contains 20 class(es) and 305 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DistributionsTestCase`, `TestDistributions`, `Dummy`, `SubClass`, `Rounded`, `ArgMax`, `ScipyCategorical`, `ScipyMixtureNormal`, `TestRsample`, `TestDistributionShapes`, `TestKL`, `Binomial30`, `TestConstraints`, `TestNumericalStability`, `TestLazyLogitsInitialization`, `TestAgainstScipy`, `TestFunctors`, `TestValidation`, `Delta`, `TestJit`

**Functions defined**: `pairwise`, `is_all_nan`, `_get_examples`, `_get_bad_examples`, `setUp`, `_gradcheck_log_prob`, `apply_fn`, `_check_forward_ad`, `_check_log_prob`, `_check_sampler_sampler`, `_check_sampler_discrete`, `_check_enumerate_support`, `test_repr`, `test_sample_detached`, `test_rsample_requires_grad`, `test_enumerate_support_type`, `test_lazy_property_grad`, `y`, `test`, `test_has_examples`

**Key imports**: math, numbers, unittest, namedtuple, product, shuffle, version, torch, torch.autograd.forward_ad as fwAD, inf, nan


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributions`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `math`
- `numbers`
- `unittest`
- `collections`: namedtuple
- `itertools`: product
- `random`: shuffle
- `packaging`: version
- `torch`
- `torch.autograd.forward_ad as fwAD`
- `torch.autograd`: grad
- `torch.autograd.functional`: jacobian
- `torch.distributions.constraint_registry`: transform_to
- `torch.distributions.constraints`: Constraint, is_dependent
- `torch.distributions.dirichlet`: _Dirichlet_backward
- `torch.distributions.kl`: _kl_expfamily_expfamily
- `torch.nn.functional`: softmax
- `torch.testing._internal.common_cuda`: TEST_CUDA
- `numpy as np`
- `scipy.special`
- `scipy.stats`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


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

This is a test file. Run it with:

```bash
python test/distributions/test_distributions.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributions`):

- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_transforms.py_docs.md`](./test_transforms.py_docs.md)
- [`test_constraints.py_docs.md`](./test_constraints.py_docs.md)


## Cross-References

- **File Documentation**: `test_distributions.py_docs.md`
- **Keyword Index**: `test_distributions.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
