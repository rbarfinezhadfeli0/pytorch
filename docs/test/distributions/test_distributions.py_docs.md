# Documentation: test_distributions.py

## File Metadata
- **Path**: `test/distributions/test_distributions.py`
- **Size**: 294438 bytes
- **Lines**: 7128
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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
        distributions_with_examples = {e.Dist for e in _get_examples()}
        for Dist in globals().values():
            if (
                isinstance(Dist, type)
                and issubclass(Dist, Distribution)
                and Dist is not Distribution
                and Dist is not ExponentialFamily
            ):
                self.assertIn(
                    Dist,
                    distributions_with_examples,
                    f"Please add {Dist.__name__} to the _get_examples list in test_distributions.py",
                )

    def test_support_attributes(self):
        for Dist, params in _get_examples():
            for param in params:
                d = Dist(**param)
                event_dim = len(d.event_shape)
                self.assertEqual(d.support.event_dim, event_dim)
                try:
                    self.assertEqual(Dist.support.event_dim, event_dim)
                except NotImplementedError:
                    pass
                is_discrete = d.support.is_discrete
                try:
                    self.assertEqual(Dist.support.is_discrete, is_discrete)
                except NotImplementedError:
                    pass

    def test_distribution_expand(self):
        shapes = [torch.Size(), torch.Size((2,)), torch.Size((2, 1))]
        for Dist, params in _get_examples():
            for param in params:
                for shape in shapes:
                    d = Dist(**param)
                    expanded_shape = shape + d.batch_shape
                    original_shape = d.batch_shape + d.event_shape
                    expected_shape = shape + original_shape
                    expanded = d.expand(batch_shape=list(expanded_shape))
                    sample = expanded.sample()
                    actual_shape = expanded.sample().shape
                    self.assertEqual(expanded.__class__, d.__class__)
                    self.assertEqual(d.sample().shape, original_shape)
                    self.assertEqual(expanded.log_prob(sample), d.log_prob(sample))
                    self.assertEqual(actual_shape, expected_shape)
                    self.assertEqual(expanded.batch_shape, expanded_shape)
                    try:
                        self.assertEqual(
                            expanded.mean, d.mean.expand(expanded_shape + d.event_shape)
                        )
                        self.assertEqual(
                            expanded.variance,
                            d.variance.expand(expanded_shape + d.event_shape),
                        )
                    except NotImplementedError:
                        pass

    def test_distribution_subclass_expand(self):
        expand_by = torch.Size((2,))
        for Dist, params in _get_examples():

            class SubClass(Dist):
                pass

            for param in params:
                d = SubClass(**param)
                expanded_shape = expand_by + d.batch_shape
                original_shape = d.batch_shape + d.event_shape
                expected_shape = expand_by + original_shape
                expanded = d.expand(batch_shape=expanded_shape)
                sample = expanded.sample()
                actual_shape = expanded.sample().shape
                self.assertEqual(expanded.__class__, d.__class__)
                self.assertEqual(d.sample().shape, original_shape)
                self.assertEqual(expanded.log_prob(sample), d.log_prob(sample))
                self.assertEqual(actual_shape, expected_shape)

    @set_default_dtype(torch.double)
    def test_bernoulli(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        r = torch.tensor(0.3, requires_grad=True)
        s = 0.3
        self.assertEqual(Bernoulli(p).sample((8,)).size(), (8, 3))
        self.assertFalse(Bernoulli(p).sample().requires_grad)
        self.assertEqual(Bernoulli(r).sample((8,)).size(), (8,))
        self.assertEqual(Bernoulli(r).sample().size(), ())
        self.assertEqual(
            Bernoulli(r).sample((3, 2)).size(),
            (
                3,
                2,
            ),
        )
        self.assertEqual(Bernoulli(s).sample().size(), ())
        self._gradcheck_log_prob(Bernoulli, (p,))

        def ref_log_prob(idx, val, log_prob):
            prob = p[idx]
            self.assertEqual(log_prob, math.log(prob if val else 1 - prob))

        self._check_log_prob(Bernoulli(p), ref_log_prob)
        self._check_log_prob(Bernoulli(logits=p.log() - (-p).log1p()), ref_log_prob)
        self.assertRaises(NotImplementedError, Bernoulli(r).rsample)

        # check entropy computation
        self.assertEqual(
            Bernoulli(p).entropy(),
            torch.tensor([0.6108, 0.5004, 0.6730]),
            atol=1e-4,
            rtol=0,
        )
        self.assertEqual(Bernoulli(torch.tensor([0.0])).entropy(), torch.tensor([0.0]))
        self.assertEqual(
            Bernoulli(s).entropy(), torch.tensor(0.6108), atol=1e-4, rtol=0
        )

        self._check_forward_ad(torch.bernoulli)
        self._check_forward_ad(lambda x: x.bernoulli_())
        self._check_forward_ad(lambda x: x.bernoulli_(x.detach().clone()))
        self._check_forward_ad(lambda x: x.bernoulli_(x))

    def test_bernoulli_enumerate_support(self):
        examples = [
            ({"probs": [0.1]}, [[0], [1]]),
            ({"probs": [0.1, 0.9]}, [[0], [1]]),
            ({"probs": [[0.1, 0.2], [0.3, 0.4]]}, [[[0]], [[1]]]),
        ]
        self._check_enumerate_support(Bernoulli, examples)

    def test_bernoulli_3d(self):
        p = torch.full((2, 3, 5), 0.5).requires_grad_()
        self.assertEqual(Bernoulli(p).sample().size(), (2, 3, 5))
        self.assertEqual(
            Bernoulli(p).sample(sample_shape=(2, 5)).size(), (2, 5, 2, 3, 5)
        )
        self.assertEqual(Bernoulli(p).sample((2,)).size(), (2, 2, 3, 5))

    @set_default_dtype(torch.double)
    def test_geometric(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        r = torch.tensor(0.3, requires_grad=True)
        s = 0.3
        self.assertEqual(Geometric(p).sample((8,)).size(), (8, 3))
        self.assertEqual(Geometric(1).sample(), 0)
        self.assertEqual(Geometric(1).log_prob(torch.tensor(1.0)), -inf)
        self.assertEqual(Geometric(1).log_prob(torch.tensor(0.0)), 0)
        self.assertFalse(Geometric(p).sample().requires_grad)
        self.assertEqual(Geometric(r).sample((8,)).size(), (8,))
        self.assertEqual(Geometric(r).sample().size(), ())
        self.assertEqual(Geometric(r).sample((3, 2)).size(), (3, 2))
        self.assertEqual(Geometric(s).sample().size(), ())
        self._gradcheck_log_prob(Geometric, (p,))
        self.assertRaises(ValueError, lambda: Geometric(0))
        self.assertRaises(NotImplementedError, Geometric(r).rsample)

        self._check_forward_ad(lambda x: x.geometric_(0.2))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @set_default_dtype(torch.double)
    def test_geometric_log_prob_and_entropy(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        s = 0.3

        def ref_log_prob(idx, val, log_prob):
            prob = p[idx].detach()
            self.assertEqual(log_prob, scipy.stats.geom(prob, loc=-1).logpmf(val))

        self._check_log_prob(Geometric(p), ref_log_prob)
        self._check_log_prob(Geometric(logits=p.log() - (-p).log1p()), ref_log_prob)

        # check entropy computation
        self.assertEqual(
            Geometric(p).entropy(),
            scipy.stats.geom(p.detach().numpy(), loc=-1).entropy(),
            atol=1e-3,
            rtol=0,
        )
        self.assertEqual(
            float(Geometric(s).entropy()),
            scipy.stats.geom(s, loc=-1).entropy().item(),
            atol=1e-3,
            rtol=0,
        )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_geometric_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for prob in [0.01, 0.18, 0.8]:
            self._check_sampler_discrete(
                Geometric(prob),
                scipy.stats.geom(p=prob, loc=-1),
                f"Geometric(prob={prob})",
            )

    @set_default_dtype(torch.double)
    def test_binomial(self):
        p = torch.arange(0.05, 1, 0.1).requires_grad_()
        for total_count in [1, 2, 10]:
            self._gradcheck_log_prob(lambda p: Binomial(total_count, p), [p])
            self._gradcheck_log_prob(
                lambda p: Binomial(total_count, None, p.log()), [p]
            )
        self.assertRaises(NotImplementedError, Binomial(10, p).rsample)

    test_binomial_half = set_default_dtype(torch.float16)(test_binomial)
    test_binomial_bfloat16 = set_default_dtype(torch.bfloat16)(test_binomial)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_binomial_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for prob in [0.01, 0.1, 0.5, 0.8, 0.9]:
            for count in [2, 10, 100, 500]:
                self._check_sampler_discrete(
                    Binomial(total_count=count, probs=prob),
                    scipy.stats.binom(count, prob),
                    f"Binomial(total_count={count}, probs={prob})",
                )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @set_default_dtype(torch.double)
    def test_binomial_log_prob_and_entropy(self):
        probs = torch.arange(0.05, 1, 0.1)
        for total_count in [1, 2, 10]:

            def ref_log_prob(idx, x, log_prob):
                p = probs.view(-1)[idx].item()
                expected = scipy.stats.binom(total_count, p).logpmf(x)
                self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

            self._check_log_prob(Binomial(total_count, probs), ref_log_prob)
            logits = probs_to_logits(probs, is_binary=True)
            self._check_log_prob(Binomial(total_count, logits=logits), ref_log_prob)

            bin = Binomial(total_count, logits=logits)
            self.assertEqual(
                bin.entropy(),
                scipy.stats.binom(
                    total_count, bin.probs.detach().numpy(), loc=-1
                ).entropy(),
                atol=1e-3,
                rtol=0,
            )

    def test_binomial_stable(self):
        logits = torch.tensor([-100.0, 100.0], dtype=torch.float)
        total_count = 1.0
        x = torch.tensor([0.0, 0.0], dtype=torch.float)
        log_prob = Binomial(total_count, logits=logits).log_prob(x)
        self.assertTrue(torch.isfinite(log_prob).all())

        # make sure that the grad at logits=0, value=0 is 0.5
        x = torch.tensor(0.0, requires_grad=True)
        y = Binomial(total_count, logits=x).log_prob(torch.tensor(0.0))
        self.assertEqual(grad(y, x)[0], torch.tensor(-0.5))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @set_default_dtype(torch.double)
    def test_binomial_log_prob_vectorized_count(self):
        probs = torch.tensor([0.2, 0.7, 0.9])
        for total_count, sample in [
            (torch.tensor([10]), torch.tensor([7.0, 3.0, 9.0])),
            (torch.tensor([1, 2, 10]), torch.tensor([0.0, 1.0, 9.0])),
        ]:
            log_prob = Binomial(total_count, probs).log_prob(sample)
            expected = scipy.stats.binom(
                total_count.cpu().numpy(), probs.cpu().numpy()
            ).logpmf(sample)
            self.assertEqual(log_prob, expected, atol=1e-4, rtol=0)

    def test_binomial_enumerate_support(self):
        examples = [
            ({"probs": [0.1], "total_count": 2}, [[0], [1], [2]]),
            ({"probs": [0.1, 0.9], "total_count": 2}, [[0], [1], [2]]),
            (
                {"probs": [[0.1, 0.2], [0.3, 0.4]], "total_count": 3},
                [[[0]], [[1]], [[2]], [[3]]],
            ),
        ]
        self._check_enumerate_support(Binomial, examples)

    @set_default_dtype(torch.double)
    def test_binomial_extreme_vals(self):
        total_count = 100
        bin0 = Binomial(total_count, 0)
        self.assertEqual(bin0.sample(), 0)
        self.assertEqual(bin0.log_prob(torch.tensor([0.0]))[0], 0, atol=1e-3, rtol=0)
        self.assertEqual(float(bin0.log_prob(torch.tensor([1.0])).exp()), 0)
        bin1 = Binomial(total_count, 1)
        self.assertEqual(bin1.sample(), total_count)
        self.assertEqual(
            bin1.log_prob(torch.tensor([float(total_count)]))[0], 0, atol=1e-3, rtol=0
        )
        self.assertEqual(
            float(bin1.log_prob(torch.tensor([float(total_count - 1)])).exp()), 0
        )
        zero_counts = torch.zeros(torch.Size((2, 2)))
        bin2 = Binomial(zero_counts, 1)
        self.assertEqual(bin2.sample(), zero_counts)
        self.assertEqual(bin2.log_prob(zero_counts), zero_counts)

    @set_default_dtype(torch.double)
    def test_binomial_vectorized_count(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        total_count = torch.tensor([[4, 7], [3, 8]], dtype=torch.float64)
        bin0 = Binomial(total_count, torch.tensor(1.0))
        self.assertEqual(bin0.sample(), total_count)
        bin1 = Binomial(total_count, torch.tensor(0.5))
        samples = bin1.sample(torch.Size((100000,)))
        self.assertTrue((samples <= total_count.type_as(samples)).all())
        self.assertEqual(samples.mean(dim=0), bin1.mean, atol=0.02, rtol=0)
        self.assertEqual(samples.var(dim=0), bin1.variance, atol=0.02, rtol=0)

    @set_default_dtype(torch.double)
    def test_negative_binomial(self):
        p = torch.arange(0.05, 1, 0.1).requires_grad_()
        for total_count in [1, 2, 10]:
            self._gradcheck_log_prob(lambda p: NegativeBinomial(total_count, p), [p])
            self._gradcheck_log_prob(
                lambda p: NegativeBinomial(total_count, None, p.log()), [p]
            )
        self.assertRaises(NotImplementedError, NegativeBinomial(10, p).rsample)
        self.assertRaises(NotImplementedError, NegativeBinomial(10, p).entropy)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_negative_binomial_log_prob(self):
        probs = torch.arange(0.05, 1, 0.1)
        for total_count in [1, 2, 10]:

            def ref_log_prob(idx, x, log_prob):
                p = probs.view(-1)[idx].item()
                expected = scipy.stats.nbinom(total_count, 1 - p).logpmf(x)
                self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

            self._check_log_prob(NegativeBinomial(total_count, probs), ref_log_prob)
            logits = probs_to_logits(probs, is_binary=True)
            self._check_log_prob(
                NegativeBinomial(total_count, logits=logits), ref_log_prob
            )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @set_default_dtype(torch.double)
    def test_negative_binomial_log_prob_vectorized_count(self):
        probs = torch.tensor([0.2, 0.7, 0.9])
        for total_count, sample in [
            (torch.tensor([10]), torch.tensor([7.0, 3.0, 9.0])),
            (torch.tensor([1, 2, 10]), torch.tensor([0.0, 1.0, 9.0])),
        ]:
            log_prob = NegativeBinomial(total_count, probs).log_prob(sample)
            expected = scipy.stats.nbinom(
                total_count.cpu().numpy(), 1 - probs.cpu().numpy()
            ).logpmf(sample)
            self.assertEqual(log_prob, expected, atol=1e-4, rtol=0)

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "CUDA and XPU not found")
    def test_zero_excluded_binomial(self):
        vals = Binomial(
            total_count=torch.tensor(1.0).to(device_type),
            probs=torch.tensor(0.9).to(device_type),
        ).sample(torch.Size((100000000,)))
        self.assertTrue((vals >= 0).all())
        vals = Binomial(
            total_count=torch.tensor(1.0).to(device_type),
            probs=torch.tensor(0.1).to(device_type),
        ).sample(torch.Size((100000000,)))
        self.assertTrue((vals < 2).all())
        vals = Binomial(
            total_count=torch.tensor(1.0).to(device_type),
            probs=torch.tensor(0.5).to(device_type),
        ).sample(torch.Size((10000,)))
        # vals should be roughly half zeroes, half ones
        assert (vals == 0.0).sum() > 4000
        assert (vals == 1.0).sum() > 4000

    def test_torch_binomial_dtype_errors(self):
        dtypes = [torch.int, torch.long, torch.short]

        for count_dtype in dtypes:
            total_count = torch.tensor([10, 10], dtype=count_dtype)
            total_prob = torch.tensor([0.5, 0.5], dtype=torch.float)

            with self.assertRaisesRegex(
                ValueError,
                "binomial only supports floating-point dtypes for count.*",
            ):
                torch.binomial(total_count, total_prob)

        for prob_dtype in dtypes:
            total_count = torch.tensor([10, 10], dtype=torch.float)
            total_prob = torch.tensor([0.5, 0.5], dtype=prob_dtype)

            with self.assertRaisesRegex(
                ValueError,
                "binomial only supports floating-point dtypes for prob.*",
            ):
                torch.binomial(total_count, total_prob)

    @set_default_dtype(torch.double)
    def test_multinomial_1d(self):
        total_count = 10
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        self.assertEqual(Multinomial(total_count, p).sample().size(), (3,))
        self.assertEqual(Multinomial(total_count, p).sample((2, 2)).size(), (2, 2, 3))
        self.assertEqual(Multinomial(total_count, p).sample((1,)).size(), (1, 3))
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, p), [p])
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, None, p.log()), [p])
        self.assertRaises(NotImplementedError, Multinomial(10, p).rsample)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @set_default_dtype(torch.double)
    def test_multinomial_1d_log_prob_and_entropy(self):
        total_count = 10
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        dist = Multinomial(total_count, probs=p)
        x = dist.sample()
        log_prob = dist.log_prob(x)
        expected = torch.tensor(
            scipy.stats.multinomial.logpmf(
                x.numpy(), n=total_count, p=dist.probs.detach().numpy()
            )
        )
        self.assertEqual(log_prob, expected)

        dist = Multinomial(total_count, logits=p.log())
        x = dist.sample()
        log_prob = dist.log_prob(x)
        expected = torch.tensor(
            scipy.stats.multinomial.logpmf(
                x.numpy(), n=total_count, p=dist.probs.detach().numpy()
            )
        )
        self.assertEqual(log_prob, expected)

        expected = scipy.stats.multinomial.entropy(
            total_count, dist.probs.detach().numpy()
        )
        self.assertEqual(dist.entropy(), expected, atol=1e-3, rtol=0)

    @set_default_dtype(torch.double)
    def test_multinomial_2d(self):
        total_count = 10
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        p = torch.tensor(probabilities, requires_grad=True)
        s = torch.tensor(probabilities_1, requires_grad=True)
        self.assertEqual(Multinomial(total_count, p).sample().size(), (2, 3))
        self.assertEqual(
            Multinomial(total_count, p).sample(sample_shape=(3, 4)).size(), (3, 4, 2, 3)
        )
        self.assertEqual(Multinomial(total_count, p).sample((6,)).size(), (6, 2, 3))
        set_rng_seed(0)
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, p), [p])
        self._gradcheck_log_prob(lambda p: Multinomial(total_count, None, p.log()), [p])

        # sample check for extreme value of probs
        self.assertEqual(
            Multinomial(total_count, s).sample(),
            torch.tensor([[total_count, 0], [0, total_count]], dtype=torch.float64),
        )

    def test_multinomial_sequential_draw(self):
        # Adapted after script mentioned in https://github.com/pytorch/pytorch/issues/132395
        torch.manual_seed(0xDE0B6B3A764007E8)
        prob = torch.ones(26)
        dups_mult = 0
        perm_counts_mult = {}
        for _ in range(300_000):
            p = tuple(torch.multinomial(prob, prob.numel(), replacement=False).tolist())
            if p in perm_counts_mult:
                dups_mult += 1
                perm_counts_mult[p] += 1
            else:
                perm_counts_mult[p] = 1
        self.assertLess(dups_mult, 10)

    @set_default_dtype(torch.double)
    def test_categorical_1d(self):
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        self.assertTrue(is_all_nan(Categorical(p).mean))
        self.assertTrue(is_all_nan(Categorical(p).variance))
        self.assertEqual(Categorical(p).sample().size(), ())
        self.assertFalse(Categorical(p).sample().requires_grad)
        self.assertEqual(Categorical(p).sample((2, 2)).size(), (2, 2))
        self.assertEqual(Categorical(p).sample((1,)).size(), (1,))
        self._gradcheck_log_prob(Categorical, (p,))
        self.assertRaises(NotImplementedError, Categorical(p).rsample)

    @set_default_dtype(torch.double)
    def test_categorical_2d(self):
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        probabilities_1 = [[1.0, 0.0], [0.0, 1.0]]
        p = torch.tensor(probabilities, requires_grad=True)
        s = torch.tensor(probabilities_1, requires_grad=True)
        self.assertEqual(Categorical(p).mean.size(), (2,))
        self.assertEqual(Categorical(p).variance.size(), (2,))
        self.assertTrue(is_all_nan(Categorical(p).mean))
        self.assertTrue(is_all_nan(Categorical(p).variance))
        self.assertEqual(Categorical(p).sample().size(), (2,))
        self.assertEqual(Categorical(p).sample(sample_shape=(3, 4)).size(), (3, 4, 2))
        self.assertEqual(Categorical(p).sample((6,)).size(), (6, 2))
        self._gradcheck_log_prob(Categorical, (p,))

        # sample check for extreme value of probs
        set_rng_seed(0)
        self.assertEqual(
            Categorical(s).sample(sample_shape=(2,)), torch.tensor([[0, 1], [0, 1]])
        )

        def ref_log_prob(idx, val, log_prob):
            sample_prob = p[idx][val] / p[idx].sum()
            self.assertEqual(log_prob, math.log(sample_prob))

        self._check_log_prob(Categorical(p), ref_log_prob)
        self._check_log_prob(Categorical(logits=p.log()), ref_log_prob)

        # check entropy computation
        self.assertEqual(
            Categorical(p).entropy(), torch.tensor([1.0114, 1.0297]), atol=1e-4, rtol=0
        )
        self.assertEqual(Categorical(s).entropy(), torch.tensor([0.0, 0.0]))
        # issue gh-40553
        logits = p.log()
        logits[1, 1] = logits[0, 2] = float("-inf")
        e = Categorical(logits=logits).entropy()
        self.assertEqual(e, torch.tensor([0.6365, 0.5983]), atol=1e-4, rtol=0)

    def test_categorical_enumerate_support(self):
        examples = [
            ({"probs": [0.1, 0.2, 0.7]}, [0, 1, 2]),
            ({"probs": [[0.1, 0.9], [0.3, 0.7]]}, [[0], [1]]),
        ]
        self._check_enumerate_support(Categorical, examples)

    @set_default_dtype(torch.double)
    def test_one_hot_categorical_1d(self):
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        self.assertEqual(OneHotCategorical(p).sample().size(), (3,))
        self.assertFalse(OneHotCategorical(p).sample().requires_grad)
        self.assertEqual(OneHotCategorical(p).sample((2, 2)).size(), (2, 2, 3))
        self.assertEqual(OneHotCategorical(p).sample((1,)).size(), (1, 3))
        self._gradcheck_log_prob(OneHotCategorical, (p,))
        self.assertRaises(NotImplementedError, OneHotCategorical(p).rsample)

    @set_default_dtype(torch.double)
    def test_one_hot_categorical_2d(self):
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        p = torch.tensor(probabilities, requires_grad=True)
        self.assertEqual(OneHotCategorical(p).sample().size(), (2, 3))
        self.assertEqual(
            OneHotCategorical(p).sample(sample_shape=(3, 4)).size(), (3, 4, 2, 3)
        )
        self.assertEqual(OneHotCategorical(p).sample((6,)).size(), (6, 2, 3))
        self._gradcheck_log_prob(OneHotCategorical, (p,))

        dist = OneHotCategorical(p)
        x = dist.sample()
        self.assertEqual(dist.log_prob(x), Categorical(p).log_prob(x.max(-1)[1]))

    def test_one_hot_categorical_enumerate_support(self):
        examples = [
            ({"probs": [0.1, 0.2, 0.7]}, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            ({"probs": [[0.1, 0.9], [0.3, 0.7]]}, [[[1, 0]], [[0, 1]]]),
        ]
        self._check_enumerate_support(OneHotCategorical, examples)

    def test_poisson_forward_ad(self):
        self._check_forward_ad(torch.poisson)

    def test_poisson_shape(self):
        rate = torch.randn(2, 3).abs().requires_grad_()
        rate_1d = torch.randn(1).abs().requires_grad_()
        self.assertEqual(Poisson(rate).sample().size(), (2, 3))
        self.assertEqual(Poisson(rate).sample((7,)).size(), (7, 2, 3))
        self.assertEqual(Poisson(rate_1d).sample().size(), (1,))
        self.assertEqual(Poisson(rate_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Poisson(2.0).sample((2,)).size(), (2,))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    @set_default_dtype(torch.double)
    def test_poisson_log_prob(self):
        rate = torch.randn(2, 3).abs().requires_grad_()
        rate_1d = torch.randn(1).abs().requires_grad_()
        rate_zero = torch.zeros([], requires_grad=True)

        def ref_log_prob(ref_rate, idx, x, log_prob):
            l = ref_rate.view(-1)[idx].detach()
            expected = scipy.stats.poisson.logpmf(x, l)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        set_rng_seed(0)
        self._check_log_prob(Poisson(rate), lambda *args: ref_log_prob(rate, *args))
        self._check_log_prob(
            Poisson(rate_zero), lambda *args: ref_log_prob(rate_zero, *args)
        )
        self._gradcheck_log_prob(Poisson, (rate,))
        self._gradcheck_log_prob(Poisson, (rate_1d,))

        # We cannot check gradients automatically for zero rates because the finite difference
        # approximation enters the forbidden parameter space. We instead compare with the
        # theoretical results.
        dist = Poisson(rate_zero)
        dist.log_prob(torch.ones_like(rate_zero)).backward()
        self.assertEqual(rate_zero.grad, torch.inf)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_poisson_sample(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        saved_dtype = torch.get_default_dtype()
        for dtype in [torch.float, torch.double, torch.bfloat16, torch.half]:
            torch.set_default_dtype(dtype)
            for rate in [0.1, 1.0, 5.0]:
                self._check_sampler_discrete(
                    Poisson(rate),
                    scipy.stats.poisson(rate),
                    f"Poisson(lambda={rate})",
                    failure_rate=1e-3,
                )
        torch.set_default_dtype(saved_dtype)

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "CUDA and XPU not found")
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_poisson_gpu_sample(self):
        set_rng_seed(1)
        for rate in [0.12, 0.9, 4.0]:
            self._check_sampler_discrete(
                Poisson(torch.tensor([rate]).to(device_type)),
                scipy.stats.poisson(rate),
                f"Poisson(lambda={rate}, {device_type})",
                failure_rate=1e-3,
            )

    @set_default_dtype(torch.double)
    def test_relaxed_bernoulli(self):
        p = torch.tensor([0.7, 0.2, 0.4], requires_grad=True)
        r = torch.tensor(0.3, requires_grad=True)
        s = 0.3
        temp = torch.tensor(0.67, requires_grad=True)
        self.assertEqual(RelaxedBernoulli(temp, p).sample((8,)).size(), (8, 3))
        self.assertFalse(RelaxedBernoulli(temp, p).sample().requires_grad)
        self.assertEqual(RelaxedBernoulli(temp, r).sample((8,)).size(), (8,))
        self.assertEqual(RelaxedBernoulli(temp, r).sample().size(), ())
        self.assertEqual(
            RelaxedBernoulli(temp, r).sample((3, 2)).size(),
            (
                3,
                2,
            ),
        )
        self.assertEqual(RelaxedBernoulli(temp, s).sample().size(), ())
        self._gradcheck_log_prob(RelaxedBernoulli, (temp, p))
        self._gradcheck_log_prob(RelaxedBernoulli, (temp, r))

        # test that rsample doesn't fail
        s = RelaxedBernoulli(temp, p).rsample()
        s.backward(torch.ones_like(s))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_rounded_relaxed_bernoulli(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]

        class Rounded:
            def __init__(self, dist):
                self.dist = dist

            def sample(self, *args, **kwargs):
                return torch.round(self.dist.sample(*args, **kwargs))

        for probs, temp in product([0.1, 0.2, 0.8], [0.1, 1.0, 10.0]):
            self._check_sampler_discrete(
                Rounded(RelaxedBernoulli(temp, probs)),
                scipy.stats.bernoulli(probs),
                f"Rounded(RelaxedBernoulli(temp={temp}, probs={probs}))",
                failure_rate=1e-3,
            )

        for probs in [0.001, 0.2, 0.999]:
            equal_probs = torch.tensor(0.5)
            dist = RelaxedBernoulli(1e10, probs)
            s = dist.rsample()
            self.assertEqual(equal_probs, s)

    @set_default_dtype(torch.double)
    def test_relaxed_one_hot_categorical_1d(self):
        p = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
        temp = torch.tensor(0.67, requires_grad=True)
        self.assertEqual(
            RelaxedOneHotCategorical(probs=p, temperature=temp).sample().size(), (3,)
        )
        self.assertFalse(
            RelaxedOneHotCategorical(probs=p, temperature=temp).sample().requires_grad
        )
        self.assertEqual(
            RelaxedOneHotCategorical(probs=p, temperature=temp).sample((2, 2)).size(),
            (2, 2, 3),
        )
        self.assertEqual(
            RelaxedOneHotCategorical(probs=p, temperature=temp).sample((1,)).size(),
            (1, 3),
        )
        self._gradcheck_log_prob(
            lambda t, p: RelaxedOneHotCategorical(t, p, validate_args=False), (temp, p)
        )

    @set_default_dtype(torch.double)
    def test_relaxed_one_hot_categorical_2d(self):
        probabilities = [[0.1, 0.2, 0.3], [0.5, 0.3, 0.2]]
        temp = torch.tensor([3.0], requires_grad=True)
        # The lower the temperature, the more unstable the log_prob gradcheck is
        # w.r.t. the sample. Values below 0.25 empirically fail the default tol.
        temp_2 = torch.tensor([0.25], requires_grad=True)
        p = torch.tensor(probabilities, requires_grad=True)
        self.assertEqual(RelaxedOneHotCategorical(temp, p).sample().size(), (2, 3))
        self.assertEqual(
            RelaxedOneHotCategorical(temp, p).sample(sample_shape=(3, 4)).size(),
            (3, 4, 2, 3),
        )
        self.assertEqual(
            RelaxedOneHotCategorical(temp, p).sample((6,)).size(), (6, 2, 3)
        )
        self._gradcheck_log_prob(
            lambda t, p: RelaxedOneHotCategorical(t, p, validate_args=False), (temp, p)
        )
        self._gradcheck_log_prob(
            lambda t, p: RelaxedOneHotCategorical(t, p, validate_args=False),
            (temp_2, p),
        )

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_argmax_relaxed_categorical(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]

        class ArgMax:
            def __init__(self, dist):
                self.dist = dist

            def sample(self, *args, **kwargs):
                s = self.dist.sample(*args, **kwargs)
                _, idx = torch.max(s, -1)
                return idx

        class ScipyCategorical:
            def __init__(self, dist):
                self.dist = dist

            def pmf(self, samples):
                new_samples = np.zeros(samples.shape + self.dist.p.shape)
                new_samples[np.arange(samples.shape[0]), samples] = 1
                return self.dist.pmf(new_samples)

        for probs, temp in product(
            [torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.2, 0.6])], [0.1, 1.0, 10.0]
        ):
            self._check_sampler_discrete(
                ArgMax(RelaxedOneHotCategorical(temp, probs)),
                ScipyCategorical(scipy.stats.multinomial(1, probs)),
                f"Rounded(RelaxedOneHotCategorical(temp={temp}, probs={probs}))",
                failure_rate=1e-3,
            )

        for probs in [torch.tensor([0.1, 0.9]), torch.tensor([0.2, 0.2, 0.6])]:
            equal_probs = torch.ones(probs.size()) / probs.size()[0]
            dist = RelaxedOneHotCategorical(1e10, probs)
            s = dist.rsample()
            self.assertEqual(equal_probs, s)

    @set_default_dtype(torch.double)
    def test_uniform(self):
        low = torch.zeros(5, 5, requires_grad=True)
        high = (torch.ones(5, 5) * 3).requires_grad_()
        low_1d = torch.zeros(1, requires_grad=True)
        high_1d = (torch.ones(1) * 3).requires_grad_()
        self.assertEqual(Uniform(low, high).sample().size(), (5, 5))
        self.assertEqual(Uniform(low, high).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Uniform(low_1d, high_1d).sample().size(), (1,))
        self.assertEqual(Uniform(low_1d, high_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Uniform(0.0, 1.0).sample((1,)).size(), (1,))

        # Check log_prob computation when value outside range
        uniform = Uniform(low_1d, high_1d, validate_args=False)
        above_high = torch.tensor([4.0])
        below_low = torch.tensor([-1.0])
        self.assertEqual(uniform.log_prob(above_high).item(), -inf)
        self.assertEqual(uniform.log_prob(below_low).item(), -inf)

        # check cdf computation when value outside range
        self.assertEqual(uniform.cdf(below_low).item(), 0)
        self.assertEqual(uniform.cdf(above_high).item(), 1)

        set_rng_seed(1)
        self._gradcheck_log_prob(Uniform, (low, high))
        self._gradcheck_log_prob(Uniform, (low, 1.0))
        self._gradcheck_log_prob(Uniform, (0.0, high))

        state = torch.get_rng_state()
        rand = low.new(low.size()).uniform_()
        torch.set_rng_state(state)
        u = Uniform(low, high).rsample()
        u.backward(torch.ones_like(u))
        self.assertEqual(low.grad, 1 - rand)
        self.assertEqual(high.grad, rand)
        low.grad.zero_()
        high.grad.zero_()

        self._check_forward_ad(lambda x: x.uniform_())

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_vonmises_sample(self):
        for loc in [0.0, math.pi / 2.0]:
            for concentration in [0.03, 0.3, 1.0, 10.0, 100.0]:
                self._check_sampler_sampler(
                    VonMises(loc, concentration),
                    scipy.stats.vonmises(loc=loc, kappa=concentration),
                    f"VonMises(loc={loc}, concentration={concentration})",
                    num_samples=int(1e5),
                    circular=True,
                )

    def test_vonmises_logprob(self):
        concentrations = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
        for concentration in concentrations:
            grid = torch.arange(0.0, 2 * math.pi, 1e-4)
            prob = VonMises(0.0, concentration).log_prob(grid).exp()
            norm = prob.mean().item() * 2 * math.pi
            self.assertLess(abs(norm - 1), 1e-3)

    @set_default_dtype(torch.double)
    def test_cauchy(self):
        loc = torch.zeros(5, 5, requires_grad=True)
        scale = torch.ones(5, 5, requires_grad=True)
        loc_1d = torch.zeros(1, requires_grad=True)
        scale_1d = torch.ones(1, requires_grad=True)
        self.assertTrue(is_all_nan(Cauchy(loc_1d, scale_1d).mean))
        self.assertEqual(Cauchy(loc_1d, scale_1d).variance, inf)
        self.assertEqual(Cauchy(loc, scale).sample().size(), (5, 5))
        self.assertEqual(Cauchy(loc, scale).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(Cauchy(loc_1d, scale_1d).sample().size(), (1,))
        self.assertEqual(Cauchy(loc_1d, scale_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(Cauchy(0.0, 1.0).sample((1,)).size(), (1,))

        set_rng_seed(1)
        self._gradcheck_log_prob(Cauchy, (loc, scale))
        self._gradcheck_log_prob(Cauchy, (loc, 1.0))
        self._gradcheck_log_prob(Cauchy, (0.0, scale))

        state = torch.get_rng_state()
        eps = loc.new(loc.size()).cauchy_()
        torch.set_rng_state(state)
        c = Cauchy(loc, scale).rsample()
        c.backward(torch.ones_like(c))
        self.assertEqual(loc.grad, torch.ones_like(scale))
        self.assertEqual(scale.grad, eps)
        loc.grad.zero_()
        scale.grad.zero_()

        self._check_forward_ad(lambda x: x.cauchy_())

    @set_default_dtype(torch.double)
    def test_halfcauchy(self):
        scale = torch.ones(5, 5, requires_grad=True)
        scale_1d = torch.ones(1, requires_grad=True)
        self.assertTrue(torch.isinf(HalfCauchy(scale_1d).mean).all())
        self.assertEqual(HalfCauchy(scale_1d).variance, inf)
        self.assertEqual(HalfCauchy(scale).sample().size(), (5, 5))
        self.assertEqual(HalfCauchy(scale).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(HalfCauchy(scale_1d).sample().size(), (1,))
        self.assertEqual(HalfCauchy(scale_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(HalfCauchy(1.0).sample((1,)).size(), (1,))

        set_rng_seed(1)
        self._gradcheck_log_prob(HalfCauchy, (scale,))
        self._gradcheck_log_prob(HalfCauchy, (1.0,))

        state = torch.get_rng_state()
        eps = scale.new(scale.size()).cauchy_().abs_()
        torch.set_rng_state(state)
        c = HalfCauchy(scale).rsample()
        c.backward(torch.ones_like(c))
        self.assertEqual(scale.grad, eps)
        scale.grad.zero_()

    @set_default_dtype(torch.double)
    def test_halfnormal(self):
        std = torch.randn(5, 5).abs().requires_grad_()
        std_1d = torch.randn(1).abs().requires_grad_()
        std_delta = torch.tensor([1e-5, 1e-5])
        self.assertEqual(HalfNormal(std).sample().size(), (5, 5))
        self.assertEqual(HalfNormal(std).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(HalfNormal(std_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(HalfNormal(std_1d).sample().size(), (1,))
        self.assertEqual(HalfNormal(0.6).sample((1,)).size(), (1,))
        self.assertEqual(HalfNormal(50.0).sample((1,)).size(), (1,))

        # sample check for extreme value of std
        set_rng_seed(1)
        self.assertEqual(
            HalfNormal(std_delta).sample(sample_shape=(1, 2)),
            torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
            atol=1e-4,
            rtol=0,
        )

        self._gradcheck_log_prob(HalfNormal, (std,))
        self._gradcheck_log_prob(HalfNormal, (1.0,))

        # check .log_prob() can broadcast.
        dist = HalfNormal(torch.ones(2, 1, 4))
        log_prob = dist.log_prob(torch.ones(3, 1))
        self.assertEqual(log_prob.shape, (2, 3, 4))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_halfnormal_logprob(self):
        std = torch.randn(5, 1).abs().requires_grad_()

        def ref_log_prob(idx, x, log_prob):
            s = std.view(-1)[idx].detach()
            expected = scipy.stats.halfnorm(scale=s).logpdf(x)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(HalfNormal(std), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_halfnormal_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for std in [0.1, 1.0, 10.0]:
            self._check_sampler_sampler(
                HalfNormal(std),
                scipy.stats.halfnorm(scale=std),
                f"HalfNormal(scale={std})",
            )

    @set_default_dtype(torch.double)
    def test_inversegamma(self):
        alpha = torch.randn(2, 3).exp().requires_grad_()
        beta = torch.randn(2, 3).exp().requires_grad_()
        alpha_1d = torch.randn(1).exp().requires_grad_()
        beta_1d = torch.randn(1).exp().requires_grad_()
        self.assertEqual(InverseGamma(alpha, beta).sample().size(), (2, 3))
        self.assertEqual(InverseGamma(alpha, beta).sample((5,)).size(), (5, 2, 3))
        self.assertEqual(InverseGamma(alpha_1d, beta_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(InverseGamma(alpha_1d, beta_1d).sample().size(), (1,))
        self.assertEqual(InverseGamma(0.5, 0.5).sample().size(), ())
        self.assertEqual(InverseGamma(0.5, 0.5).sample((1,)).size(), (1,))

        self._gradcheck_log_prob(InverseGamma, (alpha, beta))

        dist = InverseGamma(torch.ones(4), torch.ones(2, 1, 1))
        log_prob = dist.log_prob(torch.ones(3, 1))
        self.assertEqual(log_prob.shape, (2, 3, 4))

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_inversegamma_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for concentration, rate in product([2, 5], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(
                InverseGamma(concentration, rate),
                scipy.stats.invgamma(concentration, scale=rate),
                "InverseGamma()",
            )

    @set_default_dtype(torch.double)
    def test_lognormal(self):
        mean = torch.randn(5, 5, requires_grad=True)
        std = torch.randn(5, 5).abs().requires_grad_()
        mean_1d = torch.randn(1, requires_grad=True)
        std_1d = torch.randn(1).abs().requires_grad_()
        mean_delta = torch.tensor([1.0, 0.0])
        std_delta = torch.tensor([1e-5, 1e-5])
        self.assertEqual(LogNormal(mean, std).sample().size(), (5, 5))
        self.assertEqual(LogNormal(mean, std).sample((7,)).size(), (7, 5, 5))
        self.assertEqual(LogNormal(mean_1d, std_1d).sample((1,)).size(), (1, 1))
        self.assertEqual(LogNormal(mean_1d, std_1d).sample().size(), (1,))
        self.assertEqual(LogNormal(0.2, 0.6).sample((1,)).size(), (1,))
        self.assertEqual(LogNormal(-0.7, 50.0).sample((1,)).size(), (1,))

        # sample check for extreme value of mean, std
        set_rng_seed(1)
        self.assertEqual(
            LogNormal(mean_delta, std_delta).sample(sample_shape=(1, 2)),
            torch.tensor([[[math.exp(1), 1.0], [math.exp(1), 1.0]]]),
            atol=1e-4,
            rtol=0,
        )

        self._gradcheck_log_prob(LogNormal, (mean, std))
        self._gradcheck_log_prob(LogNormal, (mean, 1.0))
        self._gradcheck_log_prob(LogNormal, (0.0, std))

        # check .log_prob() can broadcast.
        dist = LogNormal(torch.zeros(4), torch.ones(2, 1, 1))
        log_prob = dist.log_prob(torch.ones(3, 1))
        self.assertEqual(log_prob.shape, (2, 3, 4))

        self._check_forward_ad(lambda x: x.log_normal_())

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_lognormal_logprob(self):
        mean = torch.randn(5, 1, requires_grad=True)
        std = torch.randn(5, 1).abs().requires_grad_()

        def ref_log_prob(idx, x, log_prob):
            m = mean.view(-1)[idx].detach()
            s = std.view(-1)[idx].detach()
            expected = scipy.stats.lognorm(s=s, scale=math.exp(m)).logpdf(x)
            self.assertEqual(log_prob, expected, atol=1e-3, rtol=0)

        self._check_log_prob(LogNormal(mean, std), ref_log_prob)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_lognormal_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        for mean, std in product([-1.0, 0.0, 1.0], [0.1, 1.0, 10.0]):
            self._check_sampler_sampler(
                LogNormal(mean, std),
                scipy.stats.lognorm(scale=math.exp(mean), s=std),
                f"LogNormal(loc={mean}, scale={std})",
            )

    @set_default_dtype(torch.double)
    def test_logisticnormal(self):
        set_rng_seed(1)  # see Note [Randomized statistical tests]
        mean = torch.randn(5, 5).requires_grad_()
        std = torch.randn(5, 5).abs().requires_grad_()
        mean_1d = torch.randn(1).requires_grad_()
        std_1d = torch.randn(1).abs().requires_grad_()
        mean_delta = torch.tensor([1.0, 0.0])
        std_delta = torch.tensor([1e-5, 1e-5])
        self.assertEqual(LogisticNormal(mean, std).sample().size(), (5, 6))
        self.assertEqual(LogisticNormal(mean, std).sample((7,)).size(), (7, 5, 6))
        self.assertEqual(LogisticNormal(mean_1d, std_1d).sample((1,)).size(), (1, 2))
        self.assertEqual(LogisticNormal(mean_1d, std_1d).sample().size(), (2,))
        self.assertEqual(LogisticNormal(0.2, 0.6).sample().size(), (2,))
        self.assertEqual(LogisticNormal(-0.7, 50.0).sample().size(), (2,))

        # sample check for extreme value of mean, std
        set_rng_seed(1)
        self.assertEqual(
            LogisticNormal(mean_delta, std_delta).sample(),
            torch.tensor(
                [
                    math.exp(1) / (1.0 + 1.0 + math.exp(1)),
                    1.0 / (1.0 + 1.0 + math.exp(1)),
                    1.0 / (1.0 + 1.0 + math.exp(1)),
                ]
            ),
            atol=1e-4,
            rtol=0,
        )

        # TODO: gradcheck seems to mutate the sample values so that the simplex
        # constraint fails by a very small margin.
        self._gradcheck_log_prob(
            lambda m, s: LogisticNormal(m, s, validate_args=False), (mean, std)
        )
        self._gradcheck_log_prob(
            lambda m, s: LogisticNormal(m, s, validate_args=False), (mean, 1.0)
        )
        self._gradcheck_log_prob(
            lambda m, s: LogisticNormal(m, s, validate_args=False), (0.0, std)
        )

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_logisticnormal_logprob(self):
        mean = torch.randn(5, 7).requires_grad_()
        std = torch.randn(5, 7).abs().requires_grad_()

        # Smoke test for now
        # TODO: Once _check_log_prob works with multidimensional distributions,
        #       add proper testing of the log probabilities.
        dist = LogisticNormal(mean, std)
        assert dist.log_prob(dist.sample()).detach().cpu().numpy().shape == (5,)

    def _get_logistic_normal_ref_sampler(self, base_dist):
        def _sampler(num_samples):
            x = base_dist.rvs(num_samples)
            offset = np.log((x.shape[-1] + 1) - np.ones_like(x).cumsum(-1))
            z = 1.0 / (1.0 + np.exp(offset - x))
            z_cumprod = np.cumprod(1 - z, axis=-1)
            y1 = np.pad(z, ((0, 0), (0, 1)), mode="constant", constant_values=1.0)
            y2 = np.pad(
                z_cumprod, ((0, 0), (1, 0)), mode="constant", constant_values=1.0
            )
            return y1 * y2

        return _sampler

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    def test_logisticnormal_sample(self):
        set_rng_seed(0)  # see Note [Randomized statistical tests]
        means = map(np.asarray, [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)])
        covs = map(np.diag, [(0.1, 0.1), (1.0, 1.0), (10.0, 10.0)])
        for mean, cov in product(means, covs):
            base_dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
            ref_dist = scipy.stats.multivariate_normal(mean=mean, cov=cov)
            ref_dist.rvs = self._get_logistic_normal_ref_sampler(base_dist)
            mean_th = torch.tensor(mean)
            std_th = torch.tensor(np.sqrt(np.diag(cov)))
            self._check_sampler_sampler(
                LogisticNormal(mean_th, std_th),
                ref_dist,
                f"LogisticNormal(loc={mean_th}, scale={std_th})",
                multivariate=True,
            )

    def test_mixture_same_family_shape(self):
        normal_case_1d = MixtureSameFamily(
            Categorical(torch.rand(5)), Normal(torch.randn(5), torch.rand(5))
        )
        normal_case_1d_batch = MixtureSameFamily(
            Categorical(torch.rand(3, 5)), Normal(torch.randn(3, 5), torch.rand(3, 5))
        )
        normal_case_1d_multi_batch = MixtureSameFamily(
            Categorical(torch.rand(4, 3, 5)),
            Normal(torch.randn(4, 3, 5), torch.rand(4, 3, 5)),
        )
        normal_case_2d = MixtureSameFamily(
            Categorical(torch.rand(5)),
            Independent(Normal(torch.randn(5, 2), torch.rand(5, 2)), 1),
        )
        normal_case_2d_batch = MixtureSameFamily(
            Categorical(torch.rand(3, 5)),
            Independent(Normal(torch.randn(3, 5, 2), torch.rand(3, 5, 2)), 1),
        )
        normal_case_2d_multi_batch = MixtureSameFamily(
            Categorical(torch.rand(4, 3, 5)),
            Independent(Normal(torch.randn(4, 3, 5, 2), torch.rand(4, 3, 5, 2)), 1),
        )

        self.assertEqual(normal_case_1d.sample().size(), ())
        self.assertEqual(normal_case_1d.sample((2,)).size(), (2,))
        self.assertEqual(normal_case_1d.sample((2, 7)).size(), (2, 7))
        self.assertEqual(normal_case_1d_batch.sample().size(), (3,))
        self.assertEqual(normal_case_1d_batch.sample((2,)).size(), (2, 3))
        self.assertEqual(normal_case_1d_batch.sample((2, 7)).size(), (2, 7, 3))
        self.assertEqual(normal_case_1d_multi_batch.sample().size(), (4, 3))
        self.assertEqual(normal_case_1d_multi_batch.sample((2,)).size(), (2, 4, 3))
        self.assertEqual(normal_case_1d_multi_batch.sample((2, 7)).size(), (2, 7, 4, 3))

        self.assertEqual(normal_case_2d.sample().size(), (2,))
        self.assertEqual(normal_case_2d.sample((2,)).size(), (2, 2))
        self.assertEqual(normal_case_2d.sample((2, 7)).size(), (2, 7, 2))
        self.a

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 20 class(es): DistributionsTestCase, TestDistributions, Dummy, SubClass, Rounded, ArgMax, ScipyCategorical, ScipyMixtureNormal, TestRsample, TestDistributionShapes, TestKL, Binomial30, TestConstraints, TestNumericalStability, TestLazyLogitsInitialization, TestAgainstScipy, TestFunctors, TestValidation, Delta, TestJit

### Functions
This file defines 305 function(s): pairwise, is_all_nan, _get_examples, _get_bad_examples, setUp, _gradcheck_log_prob, apply_fn, _check_forward_ad, _check_log_prob, _check_sampler_sampler, _check_sampler_discrete, _check_enumerate_support, test_repr, test_sample_detached, test_rsample_requires_grad, test_enumerate_support_type, test_lazy_property_grad, y, test, test_has_examples, test_support_attributes, test_distribution_expand, test_distribution_subclass_expand, test_bernoulli, ref_log_prob, test_bernoulli_enumerate_support, test_bernoulli_3d, test_geometric, test_geometric_log_prob_and_entropy, ref_log_prob


## Key Components

The file contains 19352 words across 7128 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 294438 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
