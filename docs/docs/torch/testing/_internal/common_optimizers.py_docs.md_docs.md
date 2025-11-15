# Documentation: `docs/torch/testing/_internal/common_optimizers.py_docs.md`

## File Metadata

- **Path**: `docs/torch/testing/_internal/common_optimizers.py_docs.md`
- **Size**: 53,742 bytes (52.48 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/testing/_internal/common_optimizers.py`

## File Metadata

- **Path**: `torch/testing/_internal/common_optimizers.py`
- **Size**: 85,588 bytes (83.58 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# mypy: ignore-errors

import functools
import itertools
import sys
import unittest
from copy import deepcopy
from enum import Enum
from typing import Any, Union

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import (
    Adadelta,
    Adafactor,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    ASGD,
    LBFGS,
    Muon,
    NAdam,
    Optimizer,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
    SparseAdam,
)
from torch.optim.lr_scheduler import (
    ConstantLR,
    ExponentialLR,
    LinearLR,
    PolynomialLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.testing._internal.common_device_type import tol, toleranceOverride
from torch.testing._internal.common_methods_invocations import DecorateInfo
from torch.testing._internal.common_utils import (
    _TestParametrizer,
    skipIfMPS,
    skipIfTorchDynamo,
    TEST_WITH_TORCHDYNAMO,
)
from torch.utils._foreach_utils import _get_foreach_kernels_supported_devices


CUDA_CONFIG_GPUS = ["cuda", "xpu"]


class OptimizerInput:
    """Contains args / kwargs to be passed to an optimizer constructor."""

    __slots__ = ["params", "kwargs", "desc"]

    def __init__(
        self,
        params: Union[
            list[Parameter], list[Tensor], dict[Any, Any], list[dict[str, Any]]
        ],
        kwargs: dict[str, Any],
        desc: str = "",
    ):
        # params can be a list of Tensors OR param_groups OR None
        self.params = params
        self.kwargs = kwargs
        self.desc = desc

    def __repr__(self):
        return f"params={self.params}, kwargs={self.kwargs}, desc={self.desc}"


class OptimizerErrorEnum(Enum):
    """Enumerates when an error is raised when testing optimizers."""

    CONSTRUCTION_ERROR = 0
    STEP_ERROR = 1


class ErrorOptimizerInput:
    """
    An OptimizerInput that will cause the optimizer to throw an error when constructed.
    Includes the type and string of the resulting error.
    """

    __slots__ = ["optimizer_error_input", "error_on", "error_type", "error_regex"]

    def __init__(
        self,
        optimizer_error_input,
        *,
        error_on=OptimizerErrorEnum.CONSTRUCTION_ERROR,
        error_type=RuntimeError,
        error_regex="",
    ):
        self.optimizer_error_input = optimizer_error_input
        self.error_on = error_on
        self.error_type = error_type
        self.error_regex = error_regex


class OptimizerInfo:
    """Optimizer information to be used in testing."""

    def __init__(
        self,
        optim_cls: Optimizer,  # Class object for the Optimizer under test
        *,
        # Function to generate optimizer inputs EXCLUDING params. We delegate params responsibility
        # to the test using the OptimizerInfo. OptimizerInput.params is likely None.
        # Can optionally take in device to filter out certain unsupported configs
        optim_inputs_func,
        # Tuple of lambdas to generate LRScheduler instances to run with the optimizer for the
        # LRScheduler tests like test_forloop_goes_right_direction with_lrsched.
        # We DO NOT expect to thoroughly test LRSchedulers through the optimizers, so not every
        # LRScheduler configuration will be included. See test_lrscheduler.py for that instead.
        # A few optimizers like SGD and Adam will test more LRSchedulers.
        scheduler_inputs=(
            [
                lambda opt: StepLR(opt, gamma=0.9, step_size=10),
                lambda opt: ReduceLROnPlateau(opt),
            ],
        ),
        # A subset of the global-cliquey flags (fused, foreach, differentiable) the optimizer
        # supports. See NOTE: [optimizer kwarg categories] for what global-cliquey means.
        supported_impls: tuple[str, ...] = ("foreach", "differentiable"),
        # A subset of all flags, signifying which ones were only supported after the
        # original optimizer had already been released. aka impls where we need to check BC.
        not_og_supported_flags: tuple[str, ...] = (
            "foreach",
            "differentiable",
            "maximize",
            "capturable",
        ),
        # the optim supports passing in sparse gradients as well as dense grads
        supports_sparse: bool = False,
        # the optimizer constructor supports passing in capturable as a kwarg
        has_capturable_arg: bool = False,
        # the optim only supports one config: sparse grads w/ dense params, see SparseAdam
        only_supports_sparse_grads: bool = False,
        # Tuple of (optimizer kwargs, schedulers_constructors) specifically for sparse tests,
        # with especially tuned hyperparameters. These only apply if the optimizer supports
        # sparse parameters or grads.
        metadata_for_sparse=({}, []),
        # the optim supports complex parameters
        supports_complex: bool = True,
        # whether the optimizer.step() function requires a closure to be passed
        step_requires_closure: bool = False,
        # whether the optimizer supports per-param options with parameter groups
        supports_param_groups: bool = True,
        # whether the optimizer supports parameters on multiple devices
        supports_multiple_devices: bool = True,
        skips=(),  # Indicates which tests to skip
        decorators=None,  # Additional decorators to apply to generated tests
        optim_error_inputs_func=None,  # Function to generate optim inputs that error
        supports_fused_on: tuple[str, ...] = (),
    ):
        self.optim_cls = optim_cls
        self.optim_inputs_func = optim_inputs_func
        self.scheduler_inputs = scheduler_inputs
        self.supported_impls = supported_impls
        self.not_og_supported_flags = not_og_supported_flags
        self.supports_sparse = supports_sparse
        self.has_capturable_arg = has_capturable_arg
        self.metadata_for_sparse = metadata_for_sparse
        self.only_supports_sparse_grads = only_supports_sparse_grads
        self.supports_complex = supports_complex
        self.step_requires_closure = step_requires_closure
        self.supports_param_groups = supports_param_groups
        self.supports_multiple_devices = supports_multiple_devices
        self.decorators = (
            *(decorators if decorators else []),
            *(skips if skips else []),
        )
        self.optim_error_inputs_func = optim_error_inputs_func
        self.supports_fused_on = supports_fused_on

    def get_decorators(self, test_class, test_name, device, dtype, param_kwargs):
        result = []
        for decorator in self.decorators:
            if isinstance(decorator, DecorateInfo):
                if decorator.is_active(
                    test_class, test_name, device, dtype, param_kwargs
                ):
                    result.extend(decorator.decorators)
            else:
                result.append(decorator)
        return result

    @property
    def name(self):
        return self.optim_cls.__name__


class optims(_TestParametrizer):
    """Decorator for specifying a list of optimizers over which to run a test."""

    def __init__(self, optim_info_iterable, dtypes=None):
        self.optim_info_list = list(optim_info_iterable)

        # optimizers aren't limited to be one dtype as parameters can have different dtypes
        # We default to torch.float32, but dtypes should be specified through passed in
        # parameters.
        self.dtypes = dtypes if dtypes is not None else [torch.float32]

    def _parametrize_test(self, test, generic_cls, device_cls):
        if device_cls is None:
            raise RuntimeError(
                "The @optims decorator is only intended to be used in a device-specific "
                "context; use it with instantiate_device_type_tests() instead of "
                "instantiate_parametrized_tests()"
            )

        for optim_info, dtype in itertools.product(self.optim_info_list, self.dtypes):
            # Construct the test name; device / dtype parts are handled outside.
            # See [Note: device and dtype suffix placement]
            test_name = optim_info.name

            # Construct parameter kwargs to pass to the test.
            param_kwargs = {"optim_info": optim_info, "dtype": dtype}

            try:

                @functools.wraps(test)
                def test_wrapper(*args, **kwargs):
                    return test(*args, **kwargs)

                decorator_fn = functools.partial(
                    optim_info.get_decorators,
                    generic_cls.__name__,
                    test.__name__,
                    device_cls.device_type,
                    dtype,
                )

                yield (test_wrapper, test_name, param_kwargs, decorator_fn)
            except Exception as ex:
                # Provides an error message for debugging before rethrowing the exception
                print(
                    f"Failed to instantiate {test_name} for module {optim_info.name}!"
                )
                raise ex


# Helper function for generating error inputs for all optimizers, used below.
def get_error_inputs_for_all_optims(device, dtype):
    if _get_device_type(device) == "cpu":
        # Creating 2D parameters for compatibility with Muon.
        sample_param = Parameter(torch.randn(1, 1, device=device, dtype=dtype))
        sample_param2 = Parameter(torch.randn(1, 1, device=device, dtype=dtype))
        return [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=sample_param,
                    kwargs={},
                    desc="invalid param type",
                ),
                error_type=TypeError,
                error_regex="params argument given to the optimizer should be an iterable of Tensors or dicts",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[sample_param, sample_param],
                    kwargs={},
                    desc="a param group cannot have duplicate parameters",
                ),
                error_type=UserWarning,
                error_regex=".*a parameter group with duplicate parameters.*",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[{"params": sample_param}, {"params": sample_param}],
                    kwargs={},
                    desc="duplicate parameters should not occur across param groups either",
                ),
                error_type=ValueError,
                error_regex="some parameters appear in more than one parameter group",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=torch.tensor([0.001, 0.001])),
                    desc="Tensor lr must be 1-element",
                ),
                error_type=ValueError,
                error_regex="Tensor lr must be 1-element",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[("weight", sample_param), sample_param2],
                    kwargs={},
                    desc="all optimizer params should be with/without names",
                ),
                error_type=ValueError,
                error_regex="all optimizer params should be with/without names. Some param names are missing",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[
                        {"params": [sample_param], "lr": 1e-2},
                        {"params": [("weight", sample_param2)]},
                    ],
                    kwargs={},
                    desc="all optimizer param groups should be with/without names.",
                ),
                error_type=ValueError,
                error_regex="all optimizer param groups should be with/without names. "
                "cannot add param group with names to the optimizer",
            ),
        ]
    else:
        return []


# ------------------------------------------------------------------------------------------
# NOTE: [optimizer kwarg categories]
# We categorize optimizer kwargs as 3 types:
#  1. optimizer-specific flags are like amsgrad or rho or beta, flags that are specific to
#     algorithms and thus only show up for certain optimizers. There are many of these, so I
#     do not bother gathering them all and listing them here. The converse to these would be
#     global flags that every optimizer ideally _should_ support. We break global flags into
#     2 further categories and list them all below.
#  2. global-friendly = ["lr", "weight_decay", "maximize", "capturable"]
#     global-friendly flags are global flags who play nicely with all other global flags,
#     i.e., are mutually exclusive in function. This means that any pair of the following
#     flags can be toggled at once (e.g., maximize and weight_decay). Furthermore, any of the
#     following flags theoretically can be enabled with ANY other global flag, including the
#     cliquey ones (e.g, capturable and foreach).
#  3. global-cliquey = ["foreach", "fused", "differentiable"]
#     global-cliquey flags are global flags that do NOT coexist with other cliquey flags,
#     usually because they contradict each other in function. For example, one should not flip
#     both foreach AND fused to True, because they are two differing performance optimizations
#     in which you can only opt into one.
#
# The following optim_inputs_func_* sampling functions only return constructor combinations of
# optimizer-specific and global-friendly flags. This is because we are confident they would mesh
# well with additional kwargs. On the flip side of the same coin, we reserve setting the
# global-cliquey flags to individual tests and fully expect tests to edit OptimizerInput.kwargs.


def optim_inputs_func_adadelta(device, dtype=None):
    cuda_supported_configs = [
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "capturable": True},
            desc="capturable with weight decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={"lr": torch.tensor(0.001), "capturable": True},
            desc="Tensor lr with capturable",
        ),
    ]

    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 0.01}, desc="non-default lr"),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        OptimizerInput(params=None, kwargs={"maximize": True}, desc="maximize"),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize, weight_decay",
        ),
        OptimizerInput(
            params=None, kwargs={"rho": 0.95, "weight_decay": 0.9}, desc="rho"
        ),
    ] + (cuda_supported_configs if _get_device_type(device) in CUDA_CONFIG_GPUS else [])


def optim_error_inputs_func_adadelta(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if _get_device_type(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, rho=1.1),
                    desc="rho should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid rho value: 1.1",
            ),
        ]
    return error_inputs


def optim_inputs_func_adafactor(device, dtype=None):
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "lr": 0.01},
            desc="nonzero weight_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
        OptimizerInput(
            params=None,
            kwargs={"beta2_decay": -1.0},
            desc="non-default beta2_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={"d": 1.5},
            desc="non-default clipping threshold d",
        ),
    ]


def optim_error_inputs_func_adafactor(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if _get_device_type(device) == "cpu":
        complex_param = torch.rand(2, 3, device=device, dtype=torch.complex64)
        complex_param.grad = torch.rand_like(complex_param)
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(eps=(-1e-30, 1e-3)),
                    desc="epsilon1 should be >= 0",
                ),
                error_type=ValueError,
                error_regex="epsilon1 should be >= 0",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(d=0.0),
                    desc="invalid d",
                ),
                error_type=ValueError,
                error_regex="Clipping threshold d should be >= 1",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(beta2_decay=0.8),
                    desc="invalid beta2_decay",
                ),
                error_type=ValueError,
                error_regex="beta2_decay should be <= 0",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[complex_param],
                    kwargs=dict(),
                    desc="does not support complex parameters",
                ),
                error_type=RuntimeError,
                error_regex="Adafactor does not support complex parameters",
                error_on=OptimizerErrorEnum.STEP_ERROR,
            ),
        ]
    return error_inputs


def optim_inputs_func_adagrad(device, dtype=None):
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
        OptimizerInput(params=None, kwargs={"lr": 0.1}, desc="non-default lr"),
        OptimizerInput(
            params=None,
            kwargs={"initial_accumulator_value": 0.1, "weight_decay": 0.1},
            desc="initial_accumulator_value",
        ),
        OptimizerInput(
            params=None,
            kwargs={"lr": 0.1, "lr_decay": 0.5, "weight_decay": 0.1},
            desc="lr_decay",
        ),  # TODO: Move out to testing in param_group?
        OptimizerInput(
            params=None,
            kwargs={"lr": torch.tensor(0.001)},
            desc="Tensor lr",
        ),
    ]


def optim_error_inputs_func_adagrad(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if _get_device_type(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, lr_decay=-0.5),
                    desc="lr_decay must be bigger than 0",
                ),
                error_type=ValueError,
                error_regex="Invalid lr_decay value: -0.5",
            ),
        ]
    return error_inputs


# TODO: consider tensor LR! See multi_tensor_optimizer_configs in test_optim.py --> tensor LR should work
# with all implementation code paths...
def optim_inputs_func_adam(device, dtype=None):
    cuda_supported_configs = [
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "amsgrad": True, "capturable": True},
            desc="capturable, amsgrad",
        ),
        OptimizerInput(
            params=None,
            kwargs={"lr": torch.tensor(0.001), "amsgrad": True, "capturable": True},
            desc="Tensor lr with capturable and amsgrad",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "lr": torch.tensor(0.001),
                "betas": (torch.tensor([[[0.9]]]), torch.tensor([[0.99]])),
                "amsgrad": True,
                "capturable": True,
            },
            desc="Tensor lr, Tensor betas, with capturable and amsgrad",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "lr": torch.tensor(0.001),
                "betas": (torch.tensor(0.9), torch.tensor(0.99)),
                "amsgrad": False,
                "capturable": True,
            },
            desc="Tensor lr, Tensor betas, with capturable",
        ),
    ]
    mps_supported_configs = [
        OptimizerInput(
            params=None, kwargs={"lr": torch.tensor(0.01)}, desc="Tensor lr"
        ),
    ]

    total = (
        [
            OptimizerInput(params=None, kwargs={}, desc="default"),
            OptimizerInput(params=None, kwargs={"lr": 0.01}, desc="non-default lr"),
            OptimizerInput(
                params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
            ),
            OptimizerInput(
                params=None,
                kwargs={"weight_decay": 0.1, "maximize": True},
                desc="maximize",
            ),
            OptimizerInput(
                params=None,
                kwargs={"weight_decay": 0.1, "amsgrad": True},
                desc="amsgrad",
            ),
        ]
        + (
            cuda_supported_configs
            if _get_device_type(device) in CUDA_CONFIG_GPUS
            else []
        )
        + (mps_supported_configs if _get_device_type(device) == "mps" else [])
    )
    if dtype == torch.float16:
        for input in total:
            """
            Too small eps will make denom to be zero for low precision dtype
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            For example,
            >>> a
            tensor([0.], dtype=torch.float16)
            >>> a + 1e-8
            tensor([0.], dtype=torch.float16)
            """
            input.kwargs["eps"] = 0.1
    return total


def optim_error_inputs_func_adam(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if _get_device_type(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, betas=(1.0, 0.0)),
                    desc="beta1 should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid beta parameter at index 0: 1.0",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, weight_decay=-1),
                    desc="weight_decay should > 0",
                ),
                error_type=ValueError,
                error_regex="Invalid weight_decay value: -1",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=torch.tensor(0.001), foreach=True),
                    desc="lr as Tensor doesn't work with foreach & not capturable",
                ),
                error_type=ValueError,
                error_regex="lr as a Tensor is not supported for capturable=False and foreach=True",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, betas=(0.9, torch.tensor(0.99))),
                    desc="betas must be either both floats or both Tensors",
                ),
                error_type=ValueError,
                error_regex="betas must be either both floats or both Tensors",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, betas=(torch.tensor(0.9), 0.99)),
                    desc="betas must be either both floats or both Tensors",
                ),
                error_type=ValueError,
                error_regex="betas must be either both floats or both Tensors",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(
                        lr=1e-2,
                        betas=(torch.tensor(0.9), torch.tensor(0.99)),
                        foreach=True,
                    ),
                    desc=r"betas\[0\] as a Tensor is not supported for capturable=False and foreach=True",
                ),
                error_type=ValueError,
                error_regex=r"betas\[0\] as a Tensor is not supported for capturable=False and foreach=True",
            ),
        ]
    if _get_device_type(device) in CUDA_CONFIG_GPUS:
        sample_tensor = torch.empty((), device=device, dtype=dtype)
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[sample_tensor],
                    kwargs={"foreach": True, "fused": True},
                    desc="`fused` and `foreach` cannot be `True` together",
                ),
                error_type=RuntimeError,
                error_regex="`fused` and `foreach` cannot be `True` together",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[sample_tensor],
                    kwargs={"fused": True, "differentiable": True},
                    desc="`fused` does not support `differentiable`",
                ),
                error_type=RuntimeError,
                error_regex="`fused` does not support `differentiable`",
            ),
        ]
    return error_inputs


def optim_inputs_func_adamax(device, dtype=None):
    cuda_supported_configs = [
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.9, "maximize": True, "capturable": True},
            desc="capturable, maximize, weight_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0, "maximize": True, "capturable": True},
            desc="capturable, maximize",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.9, "maximize": False, "capturable": True},
            desc="capturable, weight_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "lr": torch.tensor(0.001),
                "weight_decay": 0.9,
                "maximize": False,
                "capturable": True,
            },
            desc="capturable, weight_decay, tensor LR",
        ),
    ]

    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 0.1}, desc="non-default lr"),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=None,
            kwargs={"maximize": True},
            desc="maximize",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize, weight_decay",
        ),
    ] + (cuda_supported_configs if _get_device_type(device) in CUDA_CONFIG_GPUS else [])


def optim_error_inputs_func_adamax(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if _get_device_type(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, betas=(0.0, 1.0)),
                    desc="beta2 should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid beta parameter at index 1: 1.0",
            ),
        ]
    return error_inputs


def optim_inputs_func_adamw(device, dtype=None):
    return optim_inputs_func_adam(device, dtype)


def optim_error_inputs_func_adamw(device, dtype):
    return optim_error_inputs_func_adam(device, dtype)


def optim_inputs_func_asgd(device, dtype=None):
    cuda_supported_configs = [
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=None,
            kwargs={"maximize": True, "capturable": True},
            desc="maximize, capturable",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "capturable": True},
            desc="weight_decay, capturable",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True, "capturable": True},
            desc="maximize, weight_decay, capturable",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "lr": torch.tensor(0.001),
                "weight_decay": 0.1,
                "maximize": True,
                "capturable": True,
            },
            desc="maximize, weight_decay, capturable, tensor LR",
        ),
    ]
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lambd": 0.1}, desc="non-default lambd"),
        OptimizerInput(params=None, kwargs={"lr": 0.02}, desc="non-default lr"),
        OptimizerInput(params=None, kwargs={"t0": 100}, desc="t0"),
        OptimizerInput(params=None, kwargs={"maximize": True}, desc="maximize"),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize, nonzero weight_decay",
        ),
    ] + (cuda_supported_configs if _get_device_type(device) in CUDA_CONFIG_GPUS else [])


def optim_error_inputs_func_asgd(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if _get_device_type(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, weight_decay=-0.5),
                    desc="weight_decay should > 0",
                ),
                error_type=ValueError,
                error_regex="Invalid weight_decay value: -0.5",
            ),
        ]
    return error_inputs


def optim_inputs_func_lbfgs(device, dtype=None):
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 0.01}, desc="non-default lr"),
        OptimizerInput(
            params=None, kwargs={"lr": torch.tensor(0.001)}, desc="Tensor lr"
        ),
        OptimizerInput(
            params=None, kwargs={"tolerance_grad": 1e-6}, desc="tolerance_grad"
        ),
        OptimizerInput(
            params=None,
            kwargs={"line_search_fn": "strong_wolfe"},
            desc="strong_wolfe",
        ),
    ]


def optim_error_inputs_func_lbfgs(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    return error_inputs


def optim_inputs_func_muon(device, dtype=None):
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 0.01}, desc="non-default lr"),
        OptimizerInput(
            params=None, kwargs={"lr": torch.tensor(0.001)}, desc="Tensor lr"
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.2},
            desc="non-default weight_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={"momentum": 0.8},
            desc="non-default momentum",
        ),
        OptimizerInput(
            params=None,
            kwargs={"ns_steps": 6},
            desc="passing alternative ns_steps",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "ns_coefficients": (3.4, -4.7, 2.0),
            },
            desc="passing alternative ns_coefficients",
        ),
    ]


def optim_error_inputs_func_muon(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    complex_param = torch.rand(2, 3, device=device, dtype=torch.complex64)
    complex_param.grad = torch.rand_like(complex_param)
    non_2d_param = torch.rand(2, 3, 4, device=device, dtype=dtype)
    non_2d_param.grad = torch.rand_like(non_2d_param)
    param = torch.rand(2, 3, device=device, dtype=dtype)
    param.grad = torch.rand_like(param)
    error_inputs += [
        ErrorOptimizerInput(
            OptimizerInput(
                params=[non_2d_param],
                kwargs=dict(),
                desc="only support 2D parameters",
            ),
            error_type=ValueError,
            error_regex="Muon only supports 2D parameters",
            error_on=OptimizerErrorEnum.CONSTRUCTION_ERROR,
        ),
        ErrorOptimizerInput(
            OptimizerInput(
                params=[param],
                kwargs={"adjust_lr_fn": "arbitrary"},
                desc="only support `original` and `match_rms_adamw`",
            ),
            error_type=ValueError,
            error_regex="Adjust learning rate function arbitrary is not supported",
            error_on=OptimizerErrorEnum.CONSTRUCTION_ERROR,
        ),
        ErrorOptimizerInput(
            OptimizerInput(
                params=[complex_param],
                kwargs=dict(),
                desc="does not support complex parameters",
            ),
            error_type=RuntimeError,
            error_regex="Muon does not support complex parameters",
            error_on=OptimizerErrorEnum.STEP_ERROR,
        ),
    ]
    return error_inputs


def optim_inputs_func_nadam(device, dtype=None):
    cuda_supported_configs = [
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.9, "momentum_decay": 6e-3, "capturable": True},
            desc="weight_decay, capturable",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "weight_decay": 0.9,
                "momentum_decay": 6e-3,
                "decoupled_weight_decay": True,
                "capturable": True,
            },
            desc="decoupled_weight_decay, capturable",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "lr": torch.tensor(0.001),
                "weight_decay": 0.9,
                "momentum_decay": 6e-3,
                "decoupled_weight_decay": True,
                "capturable": True,
            },
            desc="decoupled_weight_decay, capturable",
        ),
    ]
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 1e-3}, desc="non-default lr"),
        OptimizerInput(
            params=None,
            kwargs={"momentum_decay": 6e-3},
            desc="non-zero momentum_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "weight_decay": 0.1,
            },
            desc="weight_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "momentum_decay": 6e-3},
            desc="weight_decay, momentum_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "weight_decay": 0.1,
                "momentum_decay": 6e-3,
                "decoupled_weight_decay": True,
            },
            desc="decoupled_weight_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
    ] + (cuda_supported_configs if _get_device_type(device) in CUDA_CONFIG_GPUS else [])


def optim_error_inputs_func_nadam(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if _get_device_type(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, betas=(1.0, 0.0)),
                    desc="beta1 should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid beta parameter at index 0: 1.0",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, momentum_decay=-0.2),
                    desc="momentum_decay should > 0",
                ),
                error_type=ValueError,
                error_regex="Invalid momentum_decay value: -0.2",
            ),
        ]
    return error_inputs


# Weird story bro, NAdam and RAdam do not have maximize.
def optim_inputs_func_radam(device=None, dtype=None):
    cuda_supported_configs = [
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=None,
            kwargs={
                "capturable": True,
                "weight_decay": 0.1,
            },
            desc="capturable, weight_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "capturable": True,
                "weight_decay": 0.1,
                "decoupled_weight_decay": True,
            },
            desc="capturable, weight_decay, decoupled_weight_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "lr": torch.tensor(0.001),
                "capturable": True,
                "weight_decay": 0.1,
                "decoupled_weight_decay": True,
            },
            desc="capturable, weight_decay, decoupled_weight_decay, tensor LR",
        ),
    ]
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 2e-3}, desc="non-default lr"),
        OptimizerInput(params=None, kwargs={"eps": 1e-6}, desc="non-default eps"),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "decoupled_weight_decay": True},
            desc="decoupled_weight_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
    ] + (cuda_supported_configs if _get_device_type(device) in CUDA_CONFIG_GPUS else [])


def optim_error_inputs_func_radam(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if _get_device_type(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, betas=(1.0, 0.0)),
                    desc="beta1 should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid beta parameter at index 0: 1.0",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, weight_decay=-1),
                    desc="weight_decay should > 0",
                ),
                error_type=ValueError,
                error_regex="Invalid weight_decay value: -1",
            ),
        ]
    return error_inputs


def optim_inputs_func_rmsprop(device, dtype=None):
    cuda_supported_configs = [
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True, "capturable": True},
            desc="capturable, maximize",
        ),
        OptimizerInput(
            params=None,
            kwargs={"lr": torch.tensor(0.001), "capturable": True},
            desc="Tensor lr with capturable",
        ),
    ]

    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 1e-3}, desc="non-default lr"),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.1}, desc="nonzero weight_decay"
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "maximize": True,
            },
            desc="maximize",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "centered": True},
            desc="centered",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "maximize": True,
                "weight_decay": 0.1,
            },
            desc="maximize, weight_decay",
        ),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "centered": True, "momentum": 0.1},
            desc="momentum",
        ),
        OptimizerInput(
            params=None,
            kwargs={
                "weight_decay": 0.1,
                "centered": True,
                "momentum": 0.1,
                "maximize": True,
            },
            desc="maximize, centered, weight_decay, w/ momentum",
        ),
    ] + (cuda_supported_configs if _get_device_type(device) in CUDA_CONFIG_GPUS else [])


def optim_error_inputs_func_rmsprop(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if _get_device_type(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, momentum=-1.0),
                    desc="momentum should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid momentum value: -1.0",
            ),
        ]
    return error_inputs


def optim_inputs_func_rprop(device, dtype=None):
    cuda_supported_configs = [
        OptimizerInput(params=None, kwargs={"capturable": True}, desc="capturable"),
        OptimizerInput(
            params=None,
            kwargs={"lr": torch.tensor(0.001), "capturable": True},
            desc="Tensor lr with capturable",
        ),
    ]

    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 2e-4}, desc="non-default lr"),
        OptimizerInput(
            params=None, kwargs={"etas": (0.5, 1.5)}, desc="non-default etas"
        ),
        OptimizerInput(
            params=None,
            kwargs={"step_sizes": (2e-6, 100)},
            desc="non-default step_sizes",
        ),
        OptimizerInput(params=None, kwargs={"maximize": True}, desc="maximize"),
    ] + (cuda_supported_configs if _get_device_type(device) in CUDA_CONFIG_GPUS else [])


def optim_error_inputs_func_rprop(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if _get_device_type(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, etas=(1.0, 0.5)),
                    desc="0 < eta1 < 1 < eta2",
                ),
                error_type=ValueError,
                error_regex="Invalid eta values: 1.0, 0.5",
            ),
        ]
    return error_inputs


def optim_inputs_func_sgd(device, dtype=None):
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(params=None, kwargs={"lr": 1e-2}, desc="non-default lr"),
        OptimizerInput(
            params=None, kwargs={"lr": torch.tensor(0.001)}, desc="tensor lr"
        ),
        OptimizerInput(
            params=None, kwargs={"weight_decay": 0.5}, desc="non-zero weight_decay"
        ),
        OptimizerInput(params=None, kwargs={"momentum": 0.9}, desc="momentum"),
        OptimizerInput(
            params=None,
            kwargs={"weight_decay": 0.1, "maximize": True},
            desc="maximize",
        ),
        OptimizerInput(
            params=None,
            kwargs={"momentum": 0.9, "dampening": 0.5},
            desc="dampening",
        ),
        OptimizerInput(
            params=None,
            kwargs={"momentum": 0.9, "weight_decay": 0.1},
            desc="weight_decay w/ momentum",
        ),
        OptimizerInput(
            params=None,
            kwargs={"momentum": 0.9, "nesterov": True, "weight_decay": 0.1},
            desc="nesterov",
        ),
    ]


def optim_error_inputs_func_sgd(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)
    if _get_device_type(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, momentum=-0.5),
                    desc="momentum should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid momentum value: -0.5",
            ),
        ]
    return error_inputs


def optim_inputs_func_sparseadam(device, dtype=None):
    return [
        OptimizerInput(params=None, kwargs={}, desc="default"),
        OptimizerInput(
            params=None, kwargs={"lr": 0.01}, desc="non-default lr"
        ),  # TODO: Move out to testing in param_group?
        OptimizerInput(
            params=None, kwargs={"lr": torch.tensor(0.001)}, desc="Tensor lr"
        ),
        OptimizerInput(params=None, kwargs={"maximize": True}, desc="maximize"),
    ]


def optim_error_inputs_func_sparseadam(device, dtype):
    error_inputs = get_error_inputs_for_all_optims(device, dtype)

    if _get_device_type(device) == "cpu":
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(lr=1e-2, betas=(1.0, 0.0)),
                    desc="beta1 should be between 0 and 1",
                ),
                error_type=ValueError,
                error_regex="Invalid beta parameter at index 0: 1.0",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[
                        torch.zeros(
                            3, layout=torch.sparse_coo, device=device, dtype=dtype
                        )
                    ],
                    kwargs={},
                    desc="dense params required",
                ),
                error_type=ValueError,
                error_regex="SparseAdam requires dense parameter tensors",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[
                        {
                            "params": [
                                torch.zeros(
                                    3,
                                    layout=torch.sparse_coo,
                                    device=device,
                                    dtype=dtype,
                                )
                            ]
                        }
                    ],
                    kwargs={},
                    desc="dense params required in param_groups",
                ),
                error_type=ValueError,
                error_regex="SparseAdam requires dense parameter tensors",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[torch.rand(2, 3, device=device, dtype=torch.complex64)],
                    kwargs={},
                    desc="complex not supported",
                ),
                error_type=ValueError,
                error_regex="SparseAdam does not support complex parameters",
            ),
        ]
    return error_inputs


def _get_device_type(device: Union[str, torch.device]) -> str:
    # Returns the device type as a string, e.g., "cpu" or "cuda"
    if isinstance(device, torch.device):
        device = str(device.type)
    assert isinstance(device, str)
    return device.split(":")[0]


def _get_optim_inputs_including_global_cliquey_kwargs(
    device, dtype, optim_info, skip=()
) -> list[OptimizerInput]:
    """
    Return a list of all configs for a given optimizer as a list of OptimizerInputs,
    including configs that have supported global cliquey kwargs (foreach, fused,
    differentiable) based on optim_info.supported_impls.

    The configs (optim_inputs) returned by optim_info.optim_inputs_func(...)
    intentionally do NOT include global cliquey kwargs to give flexibility to tests.
    For example, testing correctness between toggling 
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/testing/_internal`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/testing/_internal`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


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

This is a test file. Run it with:

```bash
python docs/torch/testing/_internal/common_optimizers.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/testing/_internal`):

- [`hypothesis_utils.py_kw.md_docs.md`](./hypothesis_utils.py_kw.md_docs.md)
- [`fake_config_module2.py_kw.md_docs.md`](./fake_config_module2.py_kw.md_docs.md)
- [`static_module.py_kw.md_docs.md`](./static_module.py_kw.md_docs.md)
- [`common_pruning.py_kw.md_docs.md`](./common_pruning.py_kw.md_docs.md)
- [`composite_compliance.py_kw.md_docs.md`](./composite_compliance.py_kw.md_docs.md)
- [`common_mkldnn.py_docs.md_docs.md`](./common_mkldnn.py_docs.md_docs.md)
- [`triton_utils.py_docs.md_docs.md`](./triton_utils.py_docs.md_docs.md)
- [`common_dtype.py_docs.md_docs.md`](./common_dtype.py_docs.md_docs.md)
- [`common_methods_invocations.py_docs.md_docs.md`](./common_methods_invocations.py_docs.md_docs.md)
- [`hypothesis_utils.py_docs.md_docs.md`](./hypothesis_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `common_optimizers.py_docs.md_docs.md`
- **Keyword Index**: `common_optimizers.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
