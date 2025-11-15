# Documentation: common_modules.py

## File Metadata
- **Path**: `torch/testing/_internal/common_modules.py`
- **Size**: 209310 bytes
- **Lines**: 4353
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# mypy: ignore-errors

import torch
import unittest
from copy import deepcopy
from enum import Enum
from functools import wraps, partial
from itertools import chain, product
import itertools
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDNN
from torch.testing._internal.common_dtype import (
    floating_types, floating_and_complex_types_and, get_all_fp_dtypes)
from torch.testing._internal.common_device_type import (
    _TestParametrizer, _update_param_kwargs, expectedFailureMPS, toleranceOverride, tol,
    precisionOverride, skipMeta, skipMPS)
from torch.testing._internal.common_methods_invocations import DecorateInfo
from torch.testing._internal.common_nn import (
    cosineembeddingloss_reference, cross_entropy_loss_reference, ctcloss_reference,
    hingeembeddingloss_reference, huberloss_reference, kldivloss_reference,
    marginrankingloss_reference, multimarginloss_reference, multilabelmarginloss_reference,
    nllloss_reference, nlllossNd_reference, smoothl1loss_reference, softmarginloss_reference, get_reduction)
from torch.testing._internal.common_utils import (
    freeze_rng_state, skipIfMPS, GRADCHECK_NONDET_TOL, TEST_WITH_ROCM, IS_WINDOWS,
    skipIfTorchDynamo)
from types import ModuleType
import operator

# List of all namespaces containing modules to test.
MODULE_NAMESPACES: list[ModuleType] = [
    torch.nn.modules,
    torch.ao.nn.qat.modules,
    torch.ao.nn.quantizable.modules,
    torch.ao.nn.quantized.modules,
    torch.ao.nn.quantized.modules,
]

# Modules that shouldn't be tested for one reason or another.
MODULES_TO_SKIP: set[type] = {
    torch.nn.Module,  # abstract base class
    torch.nn.Container,  # deprecated
    torch.nn.NLLLoss2d,  # deprecated
    torch.ao.nn.quantized.MaxPool2d,  # aliases to nn.MaxPool2d
    torch.ao.nn.quantized.MaxPool2d,  # aliases to nn.MaxPool2d
}

# List of all module classes to test.
MODULE_CLASSES: list[type] = [*chain.from_iterable([
    [getattr(namespace, module_name) for module_name in namespace.__all__]  # type: ignore[attr-defined]
    for namespace in MODULE_NAMESPACES])]
MODULE_CLASSES = [cls for cls in MODULE_CLASSES if cls not in MODULES_TO_SKIP]

# Dict of module class -> common name. Useful for making test names more intuitive.
# Example: torch.nn.modules.linear.Linear -> "nn.Linear"
MODULE_CLASS_NAMES: dict[type, str] = {}
for namespace in MODULE_NAMESPACES:
    for module_name in namespace.__all__:  # type: ignore[attr-defined]
        module_cls = getattr(namespace, module_name)
        namespace_name = namespace.__name__.replace('torch.', '').replace('.modules', '')

        # Deal with any aliases by preferring earlier names.
        if module_cls not in MODULE_CLASS_NAMES:
            MODULE_CLASS_NAMES[module_cls] = f'{namespace_name}.{module_name}'


# Specifies the modes (i.e. train, eval) to test over.
TrainEvalMode = Enum('TrainEvalMode', ('train_only', 'eval_only', 'train_and_eval'))


class modules(_TestParametrizer):
    """ PROTOTYPE: Decorator for specifying a list of modules over which to run a test. """

    def __init__(self, module_info_iterable, allowed_dtypes=None,
                 train_eval_mode=TrainEvalMode.train_and_eval, skip_if_dynamo=True):
        self.module_info_list = list(module_info_iterable)
        self.allowed_dtypes = set(allowed_dtypes) if allowed_dtypes is not None else None
        self.train_eval_mode = train_eval_mode
        self.skip_if_dynamo = skip_if_dynamo

    def _get_training_flags(self, module_info):
        training_flags = []
        if (self.train_eval_mode == TrainEvalMode.train_only or
                self.train_eval_mode == TrainEvalMode.train_and_eval):
            training_flags.append(True)

        if (self.train_eval_mode == TrainEvalMode.eval_only or
                self.train_eval_mode == TrainEvalMode.train_and_eval):
            training_flags.append(False)

        # If train and eval modes don't differ for the module, don't bother using more than one.
        if not module_info.train_and_eval_differ:
            training_flags = training_flags[:1]

        return training_flags

    def _parametrize_test(self, test, generic_cls, device_cls):
        if device_cls is None:
            raise RuntimeError('The @modules decorator is only intended to be used in a device-specific '
                               'context; use it with instantiate_device_type_tests() instead of '
                               'instantiate_parametrized_tests()')

        for module_info in self.module_info_list:
            dtypes = set(module_info.supported_dtypes(device_cls.device_type))
            if self.allowed_dtypes is not None:
                dtypes = dtypes.intersection(self.allowed_dtypes)

            training_flags = self._get_training_flags(module_info)
            for (training, dtype) in product(training_flags, dtypes):
                # Construct the test name; device / dtype parts are handled outside.
                # See [Note: device and dtype suffix placement]
                test_name = module_info.formatted_name
                if len(training_flags) > 1:
                    test_name += f"_{'train_mode' if training else 'eval_mode'}"

                # Construct parameter kwargs to pass to the test.
                param_kwargs = {'module_info': module_info}
                _update_param_kwargs(param_kwargs, 'dtype', dtype)
                _update_param_kwargs(param_kwargs, 'training', training)

                try:

                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        return test(*args, **kwargs)

                    if self.skip_if_dynamo and not torch.testing._internal.common_utils.TEST_WITH_TORCHINDUCTOR:
                        test_wrapper = skipIfTorchDynamo("Policy: we don't run ModuleInfo tests w/ Dynamo")(test_wrapper)

                    decorator_fn = partial(module_info.get_decorators, generic_cls.__name__,
                                           test.__name__, device_cls.device_type, dtype)

                    yield (test_wrapper, test_name, param_kwargs, decorator_fn)
                except Exception as ex:
                    # Provides an error message for debugging before rethrowing the exception
                    print(f"Failed to instantiate {test_name} for module {module_info.name}!")
                    raise ex


def get_module_common_name(module_cls):
    if module_cls in MODULE_CLASS_NAMES:
        # Example: "nn.Linear"
        return MODULE_CLASS_NAMES[module_cls]
    else:
        return module_cls.__name__


class FunctionInput:
    """ Contains args and kwargs to pass as input to a function. """
    __slots__ = ['args', 'kwargs']

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class ModuleInput:
    """ Contains args / kwargs for module instantiation + forward pass. """
    __slots__ = ['constructor_input', 'forward_input', 'desc', 'reference_fn']

    def __init__(self, constructor_input, forward_input=None, desc='', reference_fn=None):
        self.constructor_input = constructor_input  # Inputs to pass during construction
        self.forward_input = forward_input  # Inputs to pass to forward()
        self.desc = desc  # Description for this set of inputs
        self.reference_fn = reference_fn  # Reference with signature: reference_fn(module, parameters, *args, **kwargs)

        if reference_fn is not None:

            @wraps(reference_fn)
            def copy_reference_fn(m, *args, **kwargs):
                # Copy inputs to avoid undesired side effects from calling the reference.
                args, kwargs = deepcopy(args), deepcopy(kwargs)

                # Note that module parameters are passed in for convenience.
                return reference_fn(m, list(m.parameters()), *args, **kwargs)

            self.reference_fn = copy_reference_fn

class ModuleErrorEnum(Enum):
    """ Enumerates when error is raised when testing modules. """
    CONSTRUCTION_ERROR = 0
    FORWARD_ERROR = 1

class ErrorModuleInput:
    """
    A ModuleInput that will cause the operation to throw an error plus information
    about the resulting error.
    """

    __slots__ = ["module_error_input", "error_on", "error_type", "error_regex"]

    def __init__(self,
                 module_error_input,
                 *,
                 error_on=ModuleErrorEnum.CONSTRUCTION_ERROR,
                 error_type=RuntimeError,
                 error_regex):
        self.module_error_input = module_error_input
        self.error_on = error_on
        self.error_type = error_type
        self.error_regex = error_regex


class ModuleInfo:
    """ Module information to be used in testing. """

    def __init__(self,
                 module_cls,  # Class object for the module under test
                 *,
                 module_inputs_func,  # Function to generate module inputs
                 skips=(),  # Indicates which tests to skip
                 decorators=None,  # Additional decorators to apply to generated tests
                 dtypes=floating_types(),  # dtypes this function is expected to work with
                 dtypesIfMPS=(torch.float16, torch.float32,),  # dtypes this function is expected to work with on MPS
                 dtypesIfHpu=(torch.bfloat16, torch.float32,),
                 supports_gradgrad=True,  # whether the op supports second order gradients
                 gradcheck_nondet_tol=0.0,  # tolerance for nondeterminism while performing gradcheck
                 module_memformat_affects_out=False,  # whether converting module to channels last will generate
                                                      # channels last output
                 train_and_eval_differ=False,  # whether the module has differing behavior between train and eval
                 module_error_inputs_func=None,  # Function to generate module inputs that error
                 gradcheck_fast_mode=None,  # Whether to use the fast implementation for gradcheck/gradgradcheck.
                                            # When set to None, defers to the default value provided by the wrapper
                                            # function around gradcheck (testing._internal.common_utils.gradcheck)
                 ):
        self.module_cls = module_cls
        self.module_inputs_func = module_inputs_func
        self.decorators = (*(decorators if decorators else []), *(skips if skips else []))
        self.dtypes = dtypes
        self.dtypesIfMPS = dtypesIfMPS
        self.dtypesIfHpu = dtypesIfHpu
        self.supports_gradgrad = supports_gradgrad
        self.gradcheck_nondet_tol = gradcheck_nondet_tol
        self.module_memformat_affects_out = module_memformat_affects_out
        self.train_and_eval_differ = train_and_eval_differ
        self.module_error_inputs_func = module_error_inputs_func
        self.gradcheck_fast_mode = gradcheck_fast_mode
        self.is_lazy = issubclass(module_cls, torch.nn.modules.lazy.LazyModuleMixin)

    def get_decorators(self, test_class, test_name, device, dtype, param_kwargs):
        result = []
        for decorator in self.decorators:
            if isinstance(decorator, DecorateInfo):
                if decorator.is_active(test_class, test_name, device, dtype, param_kwargs):
                    result.extend(decorator.decorators)
            else:
                result.append(decorator)
        return result

    def supported_dtypes(self, device_type):
        if device_type == 'mps':
            return self.dtypesIfMPS
        elif device_type == 'hpu':
            return self.dtypesIfHpu
        else:
            return self.dtypes

    @property
    def name(self):
        return get_module_common_name(self.module_cls)

    @property
    def formatted_name(self):
        return self.name.replace('.', '_')

# Start of module inputs functions.

def module_inputs_torch_nn_Linear(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    module_inputs = [
        ModuleInput(constructor_input=FunctionInput(10, 8),
                    forward_input=FunctionInput(input=make_input((4, 10))),
                    reference_fn=lambda m, p, input: torch.mm(input, p[0].t()) + p[1].view(1, -1).expand(4, 8)),
        ModuleInput(constructor_input=FunctionInput(10, 8, bias=False),
                    forward_input=FunctionInput(make_input((4, 10))),
                    desc='no_bias',
                    reference_fn=lambda m, p, i: torch.mm(i, p[0].t())),
        ModuleInput(constructor_input=FunctionInput(3, 5),
                    forward_input=FunctionInput(make_input(3)),
                    desc='no_batch_dim',
                    reference_fn=lambda m, p, i: torch.mm(i.view(1, -1), p[0].t()).view(-1) + p[1])
    ]

    return module_inputs


def module_inputs_torch_nn_Bilinear(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def bilinear_reference_fn(m, p, x1, x2, bias=True):
        result = torch.einsum('bn,anm,bm->ba', x1, p[0], x2)
        if bias:
            if x1.shape[0] == 1:
                result = result.view(-1) + p[1]
            else:
                result = result + p[1].view(1, -1).expand(x1.shape[0], p[0].shape[0])
        return result

    module_inputs = [
        ModuleInput(constructor_input=FunctionInput(2, 3, 4),
                    forward_input=FunctionInput(make_input((8, 2)), make_input((8, 3))),
                    reference_fn=bilinear_reference_fn),
        ModuleInput(constructor_input=FunctionInput(2, 3, 4, bias=False),
                    forward_input=FunctionInput(make_input((8, 2)), make_input((8, 3))),
                    desc='no_bias',
                    reference_fn=lambda m, p, x1, x2: bilinear_reference_fn(m, p, x1, x2, bias=False)),
        ModuleInput(constructor_input=FunctionInput(2, 3, 4),
                    forward_input=FunctionInput(make_input(2), make_input(3)),
                    desc='no_batch_dim',
                    reference_fn=lambda m, p, x1, x2: bilinear_reference_fn(m, p, x1.view(1, -1), x2.view(1, -1))),
    ]

    return module_inputs


def module_inputs_torch_nn_KLDivLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_batchmean', {'reduction': 'batchmean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('log_target', {'log_target': True})
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return kldivloss_reference(i, t, **constructor_kwargs)

        input = make_input((10, 10)).log()
        target = make_input((10, 10)) if kwargs.get('log_target', False) else make_input((10, 10)).log()
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(input, target),
                        desc=desc,
                        reference_fn=reference_fn)
        )

        scalar_input = make_input(()).log()
        # FIXME(rec): scalar_target is unused, perhaps should be argument to FunctionInput?
        scalar_target = (  # noqa: F841
            make_input(()) if kwargs.get('log_target', False) else make_input(()).log()
        )
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(scalar_input, scalar_input),
                        desc='scalar_' + desc,
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_NLLLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    def make_input(shape, device=device, dtype=dtype, requires_grad=requires_grad):
        return make_tensor(shape, device=device, dtype=dtype,
                           requires_grad=False).log_softmax(dim=1).requires_grad_(requires_grad)
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_none', {'reduction': 'none'}),
        ('ignore_index', {'ignore_index': 2}),
        ('weights', {'weight': make_weight(4).abs()}),
        ('weights_ignore_index', {'weight': make_weight(4).abs(), 'ignore_index': 2}),
        ('weights_ignore_index_neg', {'weight': make_weight(4).abs(), 'ignore_index': -1})
    ]

    # TODO: Uncomment when negative weights is supported.
    # negative_weight = make_weight(10)
    # negative_weight[0] = -1
    # cases.append(('weights_negative', {'weight': negative_weight}))
    module_inputs = []
    for desc, constructor_kwargs in cases:

        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return nllloss_reference(i, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((15, 4)),
                                                    torch.empty(15, device=device).uniform_().mul(4).floor().long()),
                        desc=desc,
                        reference_fn=reference_fn)
        )

        def nd_reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return nlllossNd_reference(i, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(
                            make_input((2, 4, 5, 5)),
                            torch.empty(2, 5, 5, device=device).uniform_().mul(4).floor().long()),
                        desc=f"nd_{desc}",
                        reference_fn=nd_reference_fn)
        )

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(
                            make_input((2, 4, 5, 5, 2, 2)),
                            torch.empty(2, 5, 5, 2, 2, device=device).uniform_().mul(4).floor().long()),
                        desc=f"higher_dim_{desc}",
                        reference_fn=nd_reference_fn)
        )

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(
                            make_input((2, 4, 5)),
                            torch.empty(2, 5, device=device).uniform_().mul(4).floor().long()),
                        desc=f"3d_{desc}",
                        reference_fn=nd_reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_GaussianNLLLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('homoscedastic', {'homoscedastic': True}),
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        homoscedastic = constructor_kwargs.pop('homoscedastic', False)
        var_input = make_input(1, 3).abs() if homoscedastic else make_input(4, 1).abs()
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input(4, 3),
                                                    make_target(4, 3),
                                                    var_input),
                        desc=desc,
                        reference_fn=no_batch_dim_reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_PoissonNLLLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('full', {'full': True}),
        ('no_log_input', {'log_input': False}),
        ('full_no_log_input', {'full': True, 'log_input': False}),
    ]

    def poissonnllloss_reference_fn(i, t, log_input=True, full=False, reduction='mean', eps=1e-8):
        if log_input:
            result = i.exp() - t.mul(i)
        else:
            result = i - t.mul((i + eps).log())

        if full:
            result += (t.mul(t.log()) - t + 0.5 * (2. * math.pi * t).log()).masked_fill(t <= 1, 0)

        if reduction == 'none':
            return result
        elif reduction == 'mean':
            return result.sum() / i.numel()
        else:
            return result.sum()

    module_inputs = []
    for desc, constructor_kwargs in cases:
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return poissonnllloss_reference_fn(i, t, **constructor_kwargs)

        log_input = constructor_kwargs.get('log_input', True)
        input = make_input((2, 3, 4, 5)) if log_input else make_input((2, 3, 4, 5)).abs().add(0.001)
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(input,
                                                    make_target((2, 3, 4, 5)).floor_().abs_()),
                        desc=desc,
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_MSELoss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
    ]

    def mse_loss_reference_fn(m, p, i, t, reduction='mean'):
        if reduction == 'none':
            return (i - t).pow(2)
        elif reduction == 'mean':
            return (i - t).pow(2).sum() / i.numel()
        else:
            return (i - t).pow(2).sum()

    module_inputs = []
    for desc, constructor_kwargs in cases:
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((2, 3, 4, 5)),
                                                    make_target((2, 3, 4, 5))),
                        desc=desc,
                        reference_fn=partial(mse_loss_reference_fn, **constructor_kwargs))
        )
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input(()),
                                                    make_target(())),
                        desc=f'{desc}_scalar',
                        reference_fn=partial(mse_loss_reference_fn, **constructor_kwargs))
        )

    return module_inputs


def no_batch_dim_reference_fn(m, p, *args, **kwargs):
    """Reference function for modules supporting no batch dimensions.

    Unbatched inputs are unsqueezed to form a
    single batch input before passing them to the module.
    The output is squeezed to compare with the
    output of unbatched input to the module.

    Currently it only supports modules which return a single Tensor as output.
    You can bind the following kwargs.
    Kwargs:
        batch_first[bool] : If True, all the Tensors in `args` while be unsqueezed at dim `0` .
                        and output will be squeezed at dim `0` else dim `1` for both.
        kwargs_to_batchify[dict] : Dictionary specifying the name of the argument and dimension to unsqueeze.
                               Useful if there are few arguments whose batch dimension are different
                               from the ones selected by `batch_first`.
        is_criterion[bool] : Specify if the module is a criterion and handle the reduction for output accordingly.
    """
    def get_and_pop(key, default):
        v = kwargs.get(key, default)
        if key in kwargs:
            kwargs.pop(key)
        return v

    batch_dim = 0 if get_and_pop('batch_first', True) else 1
    kwargs_to_batchify = get_and_pop('kwargs_to_batchify', None)
    is_criterion = get_and_pop('is_criterion', False)

    if kwargs_to_batchify is not None:
        assert isinstance(kwargs_to_batchify, dict)
        for k, v in kwargs.items():
            if k in kwargs_to_batchify and v is not None:
                bdim = kwargs_to_batchify[k]
                kwargs[k] = v.unsqueeze(bdim)

    single_batch_input_args = [input.unsqueeze(batch_dim) for input in args]
    with freeze_rng_state():
        output = m(*single_batch_input_args, **kwargs).squeeze(batch_dim)

    if is_criterion:
        reduction = get_reduction(m)
        if reduction == 'none':
            return output.squeeze(0)
    return output


def no_batch_dim_reference_mha(m, p, *args, **kwargs):
    """Reference function for MultiheadAttention supporting no batch dimensions.

    Unbatched inputs are unsqueezed to form a
    single batch input before passing them to the module.
    The output is squeezed to compare with the
    output of unbatched input to the module.
    """
    batch_dim = 0 if kwargs.get('batch_first', True) else 1
    if 'batch_first' in kwargs:
        kwargs.pop('batch_first')
    if 'key_padding_mask' in kwargs and kwargs['key_padding_mask'] is not None:
        kwargs['key_padding_mask'] = kwargs['key_padding_mask'].unsqueeze(0)
    single_batch_input_args = [input.unsqueeze(batch_dim) for input in args]
    with freeze_rng_state():
        output = m(*single_batch_input_args, **kwargs)
        return (output[0].squeeze(batch_dim), output[1].squeeze(0))


def no_batch_dim_reference_rnn_gru(m, p, *args, **kwargs):
    """Reference function for RNN and GRU supporting no batch dimensions.

    Unbatched inputs are unsqueezed to form a
    single batch input before passing them to the module.
    The output is squeezed to compare with the
    output of unbatched input to the module.
    """
    if len(args) == 1:
        inp, = args
        h = None
    elif len(args) == 2:
        inp, h = args
        h = h.unsqueeze(1)

    batch_dim = 0 if kwargs['batch_first'] else 1
    kwargs.pop('batch_first')
    inp = inp.unsqueeze(batch_dim)
    single_batch_input_args = (inp, h)
    with freeze_rng_state():
        output = m(*single_batch_input_args, **kwargs)
        return (output[0].squeeze(batch_dim), output[1].squeeze(1))


def no_batch_dim_reference_lstm(m, p, *args, **kwargs):
    """Reference function for LSTM supporting no batch dimensions.

    Unbatched inputs are unsqueezed to form a
    single batch input before passing them to the module.
    The output is squeezed to compare with the
    output of unbatched input to the module.
    """
    if len(args) == 1:
        inp, = args
        h = None
    elif len(args) == 2:
        inp, h = args
        h = (h[0].unsqueeze(1), h[1].unsqueeze(1))

    batch_dim = 0 if kwargs['batch_first'] else 1
    kwargs.pop('batch_first')
    inp = inp.unsqueeze(batch_dim)
    single_batch_input_args = (inp, h)
    with freeze_rng_state():
        output = m(*single_batch_input_args, **kwargs)
        return (output[0].squeeze(batch_dim), (output[1][0].squeeze(1), output[1][1].squeeze(1)))


def no_batch_dim_reference_lstmcell(m, p, *args, **kwargs):
    """Reference function for LSTMCell supporting no batch dimensions.

    The module is passed the input and target in batched form with a single item.
    The output is squeezed to compare with the no-batch input.
    """
    inp, (h, c) = args
    single_batch_input_args = (inp.unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0)))
    with freeze_rng_state():
        output = m(*single_batch_input_args, **kwargs)
        return (output[0].squeeze(0), output[1].squeeze(0))


def generate_regression_criterion_inputs(make_input):
    return [
        ModuleInput(
            constructor_input=FunctionInput(reduction=reduction),
            forward_input=FunctionInput(make_input((4, )), make_input(4,)),
            reference_fn=partial(no_batch_dim_reference_fn, is_criterion=True),
            desc=f'no_batch_dim_{reduction}'
        ) for reduction in ['none', 'mean', 'sum']]


def module_inputs_torch_nn_AvgPool1d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(kernel_size=2),
                    forward_input=FunctionInput(make_input((3, 6))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn),
        ModuleInput(constructor_input=FunctionInput(2),
                    forward_input=FunctionInput(make_input((2, 3, 6)))),
        ModuleInput(constructor_input=FunctionInput((2,), (2,)),
                    forward_input=FunctionInput(make_input((2, 3, 6))),
                    desc='stride'),
        ModuleInput(constructor_input=FunctionInput(2, 2, 1),
                    forward_input=FunctionInput(make_input((2, 3, 6))),
                    desc='stride_pad')]


def module_inputs_torch_nn_AvgPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput((2, 2)),
                    forward_input=FunctionInput(make_input((3, 6, 6))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn),
        ModuleInput(constructor_input=FunctionInput((2, 2)),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6)))),
        ModuleInput(constructor_input=FunctionInput((2, 2), (2, 2)),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='stride'),
        ModuleInput(constructor_input=FunctionInput((2, 2), (2, 2), (1, 1)),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='stride_pad'),
        ModuleInput(constructor_input=FunctionInput((2, 2), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='divisor'),
        ModuleInput(constructor_input=FunctionInput((2, 2), (2, 2), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='divisor_stride'),
        ModuleInput(constructor_input=FunctionInput((2, 2), (2, 2), (1, 1), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='divisor_stride_pad')]



def module_inputs_torch_nn_AvgPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput((2, 2, 2)),
                    forward_input=FunctionInput(make_input((3, 4, 4, 4))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn),
        ModuleInput(constructor_input=FunctionInput((2, 2, 2)),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4)))),
        ModuleInput(constructor_input=FunctionInput(2, (2, 2, 2)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),
                    desc='stride'),
        ModuleInput(constructor_input=FunctionInput(2, 2, (1, 1, 1)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),
                    desc='stride_pad'),
        ModuleInput(constructor_input=FunctionInput(4, 2, (1, 2, 1)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),
                    desc='stride_pad_gpu_fixedkw_output'),
        ModuleInput(constructor_input=FunctionInput((2, 4, 8), 1, (1, 1, 2)),
                    forward_input=FunctionInput(make_input((2, 3, 2, 4, 8))),
                    desc='stride_pad_gpu_general_output'),
        ModuleInput(constructor_input=FunctionInput(3, 1, 0),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='stride1_pad0_gpu_input'),
        ModuleInput(constructor_input=FunctionInput(2, 2, (1, 1, 1)),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='stride_pad_gpu_input_nooverlap'),
        ModuleInput(constructor_input=FunctionInput((2, 2, 2), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='divisor'),
        ModuleInput(constructor_input=FunctionInput(2, (2, 2, 2), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),
                    desc='divisor_stride'),
        ModuleInput(constructor_input=FunctionInput(2, 2, (1, 1, 1), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),
                    desc='divisor_stride_pad'),
        ModuleInput(constructor_input=FunctionInput(4, 2, (1, 2, 1), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 5, 5, 5))),
                    desc='divisor_stride_pad_gpu_fixedkw_output'),
        ModuleInput(constructor_input=FunctionInput((2, 4, 8), 1, (1, 1, 2), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 2, 4, 8))),
                    desc='divisor_stride_pad_gpu_general_output'),
        ModuleInput(constructor_input=FunctionInput(3, 1, 0, divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='divisor_stride1_pad0_gpu_input'),
        ModuleInput(constructor_input=FunctionInput(2, 2, (1, 1, 1), divisor_override=1),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='divisor_stride_pad_gpu_input_nooverlap')]



def module_inputs_torch_nn_AdaptiveAvgPool1d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((1, 3, 5))),
                    desc='single'),
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((3, 5))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(1,),
                    forward_input=FunctionInput(make_input((1, 3, 5))),
                    desc='one_output')]


def module_inputs_torch_nn_AdaptiveAvgPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='single'),
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((3, 5, 6))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(1,),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='single_1x1output'),
        ModuleInput(constructor_input=FunctionInput((3, 4)),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='tuple'),
        ModuleInput(constructor_input=FunctionInput((3, None)),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='tuple_none')]

def module_inputs_torch_nn_AdaptiveAvgPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((2, 3, 5, 2, 7))),
                    desc='single'),
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((3, 5, 2, 7))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput((3, 4, 5)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 3, 7))),
                    desc='tuple'),
        ModuleInput(constructor_input=FunctionInput((None, 4, 5)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 3, 7))),
                    desc='tuple_none'),
        ModuleInput(constructor_input=FunctionInput((3, 2, 2)),
                    forward_input=FunctionInput(make_input((1, 1, 3, 2, 6))),
                    desc='last_dim')]


def module_inputs_torch_nn_AdaptiveMaxPool1d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((1, 3, 5))),
                    desc='single'),
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((3, 5))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_AdaptiveMaxPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='single'),
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((3, 5, 6))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput((3, 4)),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='tuple'),
        ModuleInput(constructor_input=FunctionInput((3, None)),
                    forward_input=FunctionInput(make_input((1, 3, 5, 6))),
                    desc='tuple_none')]


def module_inputs_torch_nn_AdaptiveMaxPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((2, 3, 5, 6, 7))),
                    desc='single'),
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((3, 5, 6, 7))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput((3, 4, 5)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 6, 7))),
                    desc='tuple'),
        ModuleInput(constructor_input=FunctionInput((3, None, 5)),
                    forward_input=FunctionInput(make_input((2, 3, 5, 6, 7))),
                    desc='tuple_none'),
        ModuleInput(constructor_input=FunctionInput(3),
                    forward_input=FunctionInput(make_input((2, 3, 12, 9, 3))),
                    desc='single_nonatomic'),
        ModuleInput(constructor_input=FunctionInput((3, 4, 5)),
                    forward_input=FunctionInput(make_input((2, 3, 6, 4, 10))),
                    desc='tuple_nonatomic')]


def module_inputs_torch_nn_BatchNorm1d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(10,),
                    forward_input=FunctionInput(make_input((4, 10))),
                    desc='affine'),
        ModuleInput(constructor_input=FunctionInput(5,),
                    forward_input=FunctionInput(make_input((4, 5, 3))),
                    desc='3d_input'),
        ModuleInput(constructor_input=FunctionInput(10, 1e-3, None),
                    forward_input=FunctionInput(make_input((4, 10))),
                    desc='affine_simple_average'),
        ModuleInput(constructor_input=FunctionInput(10, 1e-3, 0.3, False),
                    forward_input=FunctionInput(make_input((4, 10))),
                    desc='not_affine'),
        ModuleInput(constructor_input=FunctionInput(10, 1e-3, 0.3, True, False),
                    forward_input=FunctionInput(make_input((4, 10))),
                    desc='not_tracking_stats'),
        ModuleInput(constructor_input=FunctionInput(5, 1e-3, 0.3, False),
                    forward_input=FunctionInput(make_input((4, 5, 3))),
                    desc='3d_input_not_affine'),
        ModuleInput(constructor_input=FunctionInput(5, 1e-3, 0.3, False),
                    forward_input=FunctionInput(make_input((0, 5, 9))),
                    desc='zero_batch')]


def module_inputs_torch_nn_BatchNorm2d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6)))),
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, None),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='2d_simple_average'),
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, 0.8),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='momentum'),
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, 0.8, False),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='not_affine'),
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, 0.8, True, False),
                    forward_input=FunctionInput(make_input((2, 3, 6, 6))),
                    desc='not_tracking_stats'),
        ModuleInput(constructor_input=FunctionInput(5, 1e-3, 0.3, False),
                    forward_input=FunctionInput(make_input((0, 5, 2, 2))),
                    desc='zero_batch')]


def module_inputs_torch_nn_BatchNorm3d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(3,),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4)))),
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, None),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='3d_simple_average'),
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, 0.7),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='momentum'),
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, 0.7, False),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='not_affine'),
        ModuleInput(constructor_input=FunctionInput(3, 1e-3, 0.7, True, False),
                    forward_input=FunctionInput(make_input((2, 3, 4, 4, 4))),
                    desc='not_tracking_stats'),
        ModuleInput(constructor_input=FunctionInput(5, 1e-3, 0.3, False),
                    forward_input=FunctionInput(make_input((0, 5, 2, 2, 2))),
                    desc='zero_batch')]


def module_error_inputs_torch_nn_BatchNorm1d_2d_3d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    if module_info.module_cls == torch.nn.BatchNorm1d:
        input_shape = (2, 10)
    elif module_info.module_cls == torch.nn.BatchNorm2d:
        input_shape = (2, 10, 5, 5)
    else:
        input_shape = (2, 10, 4, 4, 4)

    return [
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, eps=-1.0),
                forward_input=FunctionInput(make_input(input_shape)),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=ValueError,
            error_regex="eps must be positive"
        ),
        ErrorModuleInput(
            ModuleInput(
                constructor_input=FunctionInput(10, eps=0.0),
                forward_input=FunctionInput(make_input(input_shape)),
            ),
            error_on=ModuleErrorEnum.FORWARD_ERROR,
            error_type=ValueError,
            error_regex="eps must be positive"
        ),
    ]


def module_inputs_torch_nn_ConvNd(module_info, device, dtype, requires_grad, training, **kwargs):
    N = kwargs['N']
    lazy = kwargs.get('lazy', False)
    transposed = kwargs.get('transposed', False)
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    conv_kwargs_list = [{}] if transposed else [{}, {'padding': 'same'}]
    kernel_size, C_in, C_out = 3, 4, 5
    input_no_batch_shape = (C_in,) + tuple(i + 3 for i in range(N))
    input_batch_shape = (2,) + input_no_batch_shape
    return [
        ModuleInput(constructor_input=(FunctionInput(C_out, kernel_size, **conv_kwargs) if lazy else
                                       FunctionInput(C_in, C_out, kernel_size, **conv_kwargs)),
                    forward_input=FunctionInput(make_input(
                        input_batch_shape if with_batch else input_no_batch_shape)),
                    desc=('' if with_batch else 'no_batch_dim'),
                    reference_fn=(None if with_batch else no_batch_dim_reference_fn))
        for with_batch, conv_kwargs in itertools.product([True, False], conv_kwargs_list)
    ]


def module_inputs_torch_nn_CosineEmbeddingLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('margin', {'margin': 0.7})
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        def reference_fn(m, p, i1, i2, t, constructor_kwargs=constructor_kwargs):
            return cosineembeddingloss_reference(i1, i2, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((15, 10)), make_input((15, 10)),
                                                    make_target((15,)).sign()),
                        desc=desc,
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_ELU(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    reference_fn=lambda m, p, i: torch.where(i >= 0, i, 2 * (i.exp() - 1))),
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3,))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn),
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input((2, 3, 2, 5))),
                    desc='4d_input')]


def module_inputs_torch_nn_CELU(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    reference_fn=lambda m, p, i: torch.where(i >= 0, i, 2. * ((.5 * i).exp() - 1))),
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, i: torch.where(i >= 0, i, 2. * ((.5 * i).exp() - 1)),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(alpha=2.),
                    forward_input=FunctionInput(make_input((3,))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn)]


def module_inputs_torch_nn_GLU(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((5, 6)))),
        ModuleInput(constructor_input=FunctionInput(1),
                    forward_input=FunctionInput(make_input((5, 6, 7))),
                    desc='dim'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((4,))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn)]


def module_inputs_torch_nn_GELU(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput('none'),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, x, *_: x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput('none'),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    reference_fn=lambda m, p, x, *_: x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3,))),
                    desc='no_batch_dim',
                    reference_fn=no_batch_dim_reference_fn)]


def module_inputs_torch_nn_ReLU(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                    desc='channels_last_mem_format'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 3, 4, 5))),
                    desc='channels_last_3d_mem_format')]


def module_inputs_torch_nn_ReLU6(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                    desc='channels_last_mem_format'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 3, 4, 5))),
                    desc='channels_last_3d_mem_format')]


def module_inputs_torch_nn_LeakyReLU(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3, 2, 5)))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(0.5),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    desc='with_negval'),
        ModuleInput(constructor_input=FunctionInput(0.0),
                    forward_input=FunctionInput(make_input((10, 10))),
                    desc='with_zero_negval'),
        ModuleInput(constructor_input=FunctionInput(0.5),
                    forward_input=FunctionInput(make_input(())),
                    desc='with_negval_scalar')]


def module_inputs_torch_nn_PReLU(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4))),
                    reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
                    desc='1d'),
        ModuleInput(constructor_input=FunctionInput(3),
                    forward_input=FunctionInput(make_input((2, 3, 4))),
                    reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
                    desc='1d_multiparam'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                    reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
                    desc='2d'),
        ModuleInput(constructor_input=FunctionInput(3),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                    reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
                    desc='2d_multiparam'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5, 6))),
                    reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
                    desc='3d'),
        ModuleInput(constructor_input=FunctionInput(3),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5, 6))),
                    reference_fn=lambda m, p, i: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
                    desc='3d_multiparam')]


def module_inputs_torch_nn_SELU(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3, 2, 5)))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar')]


def module_inputs_torch_nn_SiLU(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, x, *_: x * torch.sigmoid(x),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((5, 6, 7))),
                    reference_fn=lambda m, p, x, *_: x * torch.sigmoid(x))]


def module_inputs_torch_nn_Softmax(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(1),
                    forward_input=FunctionInput(make_input((10, 20))),
                    reference_fn=lambda m, p, i: torch.exp(i).div(torch.exp(i).sum(1, True).expand(10, 20))),
        ModuleInput(constructor_input=FunctionInput(0),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, i: torch.exp(i).div(torch.exp(i).sum(0, True)),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(-1),
                    forward_input=FunctionInput(make_input((4, 5))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Softmax2d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((1, 3, 10, 20))),
                    reference_fn=lambda m, p, i: torch.exp(i).div(torch.exp(i).sum(1, False))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3, 4, 5))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_LogSoftmax(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(1),
                    forward_input=FunctionInput(make_input((10, 20))),
                    reference_fn=lambda m, p, i: torch.exp(i).div_(torch.exp(i).sum(1, True).expand(10, 20)).log_()),
        ModuleInput(constructor_input=FunctionInput(1),
                    forward_input=FunctionInput(make_input((1, 3, 10, 20))),
                    reference_fn=lambda m, p, i: torch.exp(i).div_(torch.exp(i).sum(1, False)).log_(),
                    desc='multiparam'),
        ModuleInput(constructor_input=FunctionInput(0),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, i: torch.exp(i).div_(torch.exp(i).sum(0, False)).log_(),
                    desc='multiparam_scalar'),
        ModuleInput(constructor_input=FunctionInput(-1),
                    forward_input=FunctionInput(make_input((4, 5))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Softmin(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(1),
                    forward_input=FunctionInput(make_input((10, 20)))),
        ModuleInput(constructor_input=FunctionInput(1),
                    forward_input=FunctionInput(make_input((2, 3, 5, 10))),
                    desc='multidim'),
        ModuleInput(constructor_input=FunctionInput(0),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(-1),
                    forward_input=FunctionInput(make_input((3, 4, 10))),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Softplus(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((10, 20))),
                    reference_fn=lambda m, p, i: torch.log1p(torch.exp(i))),
        ModuleInput(constructor_input=FunctionInput(2),
                    forward_input=FunctionInput(make_input((10, 20))),
                    reference_fn=lambda m, p, i: 1. / 2. * torch.log1p(torch.exp(2 * i)),
                    desc='beta'),
        ModuleInput(constructor_input=FunctionInput(2, -100),
                    forward_input=FunctionInput(make_input((10, 20))),
                    reference_fn=(
                        lambda m, p, i: ((i * 2) > -100).type_as(i) * i
                        + ((i * 2) <= -100).type_as(i) * 1. / 2. * torch.log1p(torch.exp(2 * i))),
                    desc='beta_threshold'),
        ModuleInput(constructor_input=FunctionInput(2, -100),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=(
                        lambda m, p, i: ((i * 2) > -100).type_as(i) * i
                        + ((i * 2) <= -100).type_as(i) * 1. / 2. * torch.log1p(torch.exp(2 * i))),
                    desc='beta_threshold_scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Softshrink(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3, 2, 5)))),
        ModuleInput(constructor_input=FunctionInput(1,),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    desc='lambda'),
        ModuleInput(constructor_input=FunctionInput(1,),
                    forward_input=FunctionInput(make_input(())),
                    desc='lambda_scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Softsign(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((3, 2, 5))),
                    reference_fn=lambda m, p, i: i.div(1 + torch.abs(i))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, i: i.div(1 + torch.abs(i)),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Tanh(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5)))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]



def module_inputs_torch_nn_Tanhshrink(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5)))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Threshold(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(2., 1.),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                    desc='threshold_value'),
        ModuleInput(constructor_input=FunctionInput(2., 10.),
                    forward_input=FunctionInput(make_input((2, 3, 4, 5))),
                    desc='large_value'),
        ModuleInput(constructor_input=FunctionInput(2., 1.),
                    forward_input=FunctionInput(make_input(())),
                    desc='threshold_value_scalar'),
        ModuleInput(constructor_input=FunctionInput(2., 1.),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_Mish(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((5, 6, 7))),
                    reference_fn=lambda m, p, i: i * torch.tanh(F.softplus(i))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(())),
                    reference_fn=lambda m, p, i: i * torch.tanh(F.softplus(i)),
                    desc='scalar'),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(4)),
                    reference_fn=no_batch_dim_reference_fn,
                    desc='no_batch_dim')]


def module_inputs_torch_nn_L1Loss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input((2, 3, 4)),
                                                make_input((2, 3, 4))),
                    reference_fn=lambda m, p, i, t: 1. / i.numel() * sum((a - b).abs().sum()
                                                                         for a, b in zip(i, t, strict=True))),
        ModuleInput(constructor_input=FunctionInput(),
                    forward_input=FunctionInput(make_input(()), make_input(())),
                    reference_fn=lambda m, p, i, t: 1. / i.numel() * (i - t).abs().sum(),
                    desc='scalar')] + generate_regression_criterion_inputs(make_input)


def module_inputs_torch_nn_SmoothL1Loss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)


    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return smoothl1loss_reference(i, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((5, 10)),
                                                    make_input((5, 10))),
                        desc=desc,
                        reference_fn=reference_fn)
        )
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input(()),
                                                    make_input(())),
                        desc=f'scalar_{desc}',
                        reference_fn=reference_fn)
        )

    return module_inputs



def module_inputs_torch_nn_BCELoss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('weights', {'weight': make_weight((10,))}),
    ]

    def bce_loss_reference_fn(m, p, i, t, reduction='mean', weight=None):
        result = -(t * i.log() + (1 - t) * (1 - i).log())

        if weight is not None:
            result = result * weight

        if reduction == 'none':
            return result
        elif reduction == 'mean':
            return result.sum() / i.numel()
        else:
            return result.sum()

    module_inputs = []
    for desc, constructor_kwargs in cases:
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((15, 10), low=1e-2, high=1 - 1e-2),
                                                    make_target((15, 10)).gt(0).to(dtype)),
                        desc=desc,
                        reference_fn=partial(bce_loss_reference_fn, **constructor_kwargs))
        )

    scalar_weight = make_weight(())
    module_inputs.append(
        ModuleInput(constructor_input=FunctionInput(weight=scalar_weight),
                    forward_input=FunctionInput(make_input((), low=1e-2, high=1 - 1e-2),
                                                make_target(()).gt(0).to(dtype)),
                    desc='scalar_weight',
                    reference_fn=partial(bce_loss_reference_fn, weight=scalar_weight))
    )

    return module_inputs


def module_inputs_torch_nn_BCEWithLogitsLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('weights', {'weight': make_weight((10,))}),
        ('scalar_weights', {'weight': make_weight(())})
    ]

    def bce_withlogitsloss_reference_fn(m, p, i, t, reduction='mean', weight=None):
        # TODO: add pos_weight to the definition here and corresponding SampleInputs
        max_val = (-i).clamp(min=0)
        result = (1 - t).mul_(i).add_(max_val).add_((-max_val).exp_().add_((-i - max_val).exp_()).log_())

        if weight is not None:
            result = result * weight

        if reduction == 'none':
            return result
        elif reduction == 'mean':
            return result.sum() / i.numel()
        else:
            return result.sum()

    module_inputs = []
    for desc, constructor_kwargs in cases:
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((15, 10), low=1e-2, high=1 - 1e-2),
                                                    make_target((15, 10)).gt(0).to(dtype)),
                        desc=desc,
                        reference_fn=partial(bce_withlogitsloss_reference_fn, **constructor_kwargs))
        )

    return module_inputs


def module_inputs_torch_nn_CrossEntropyLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=torch.long, requires_grad=False)
    make_weight = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    reductions: list[str] = ['mean', 'sum', 'none']
    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('weights', {'weight': make_weight((3,))}),
        ('ignore_index', {'ignore_index': 1}),
        ('label_smoothing', {'label_smoothing': 0.15}),
        ('ignore_index_label_smoothing', {'ignore_index': 1, 'label_smoothing': 0.15})
    ]

    module_inputs = []
    for reduction, (desc, constructor_kwargs) in product(reductions, cases):
        def reference_fn(m, p, i, t, reduction=reduction, constructor_kwargs=constructor_kwargs):
            return cross_entropy_loss_reference(i, t, reduction=reduction, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                        forward_input=FunctionInput(make_input((2, 3, 5, 5)),
                                                    make_target((2, 5, 5), low=0, high=3)),
                        desc=f"4d_{desc}_{reduction}",
                        reference_fn=reference_fn)
        )
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                        forward_input=FunctionInput(make_input((2, 3, 5)),
                                                    make_target((2, 5), low=0, high=3)),
                        desc=f"3d_{desc}_{reduction}",
                        reference_fn=reference_fn)
        )
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                        forward_input=FunctionInput(make_input((2, 3)),
                                                    make_target((2), low=0, high=3)),
                        desc=f"2d_{desc}_{reduction}",
                        reference_fn=reference_fn)
        )
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                        forward_input=FunctionInput(make_input((2, 3, 5, 5, 2, 2)),
                                                    make_target((2, 5, 5, 2, 2), low=0, high=3)),
                        desc=f"higher_dim_{desc}_{reduction}",
                        reference_fn=reference_fn)
        )

        if constructor_kwargs.get('ignore_index', None) is None:
            module_inputs.append(
                ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                            forward_input=FunctionInput(make_input((5, 3, 4, 2)),
                                                        make_input((5, 3, 4, 2)).softmax(dim=1)),
                            desc=f"4d_prob_target_{desc}_{reduction}",
                            reference_fn=reference_fn)
            )
            module_inputs.append(
                ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                            forward_input=FunctionInput(make_input((5, 3, 4)),
                                                        make_input((5, 3, 4)).softmax(dim=1)),
                            desc=f"3d_prob_target_{desc}_{reduction}",
                            reference_fn=reference_fn)
            )
            module_inputs.append(
                ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                            forward_input=FunctionInput(make_input((5, 3)),
                                                        make_input((5, 3)).softmax(dim=1)),
                            desc=f"2d_prob_target_{desc}_{reduction}",
                            reference_fn=reference_fn)
            )
            module_inputs.append(
                ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                            forward_input=FunctionInput(make_input((2, 3, 5, 5, 2, 2)),
                                                        make_input((2, 3, 5, 5, 2, 2)).softmax(dim=1)),
                            desc=f"higher_dim_prob_target_{desc}_{reduction}",
                            reference_fn=reference_fn)
            )
            module_inputs.append(
                ModuleInput(constructor_input=FunctionInput(reduction=reduction, **constructor_kwargs),
                            forward_input=FunctionInput(make_input((3,)),
                                                        make_target((), low=0, high=3)),
                            desc=f"no_batch_dim_{desc}_{reduction}",
                            reference_fn=partial(no_batch_dim_reference_fn, is_criterion=True))
            )

    return module_inputs



def module_inputs_torch_nn_CTCLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('blank', {'blank': 14})
    ]
    target_dtypes = [torch.int, torch.long]

    module_inputs = []
    for target_dtype, (desc, constructor_kwargs) in product(target_dtypes, cases):
        def reference_fn(m, p, i, t, il, tl, constructor_kwargs=constructor_kwargs):
            return ctcloss_reference(i, t, il, tl, **constructor_kwargs)

        blank = constructor_kwargs.get('blank', 0)
        low = 0 if blank == 14 else 1
        high = 14 if blank == 14 else 15

        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),
                forward_input=FunctionInput(make_input((50, 3, 15)).log_softmax(2),
                                            make_target((3, 30), dtype=target_dtype, low=low, high=high),
                                            (50, 50, 50), (30, 25, 20)),
                desc=f'{desc}_lengths_intlists',
                reference_fn=reference_fn)
        )
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),
                forward_input=FunctionInput(make_input((50, 3, 15)).log_softmax(2),
                                            make_target((3, 30), dtype=target_dtype, low=low, high=high),
                                            torch.tensor((50, 50, 50), device=device),
                                            torch.tensor((30, 25, 20), device=device)),
                desc=f'{desc}_lengths_tensors',
                reference_fn=reference_fn)
        )
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),
                forward_input=FunctionInput(make_input((50, 3, 15)).log_softmax(2),
                                            make_target((30 + 25 + 20,), dtype=target_dtype, low=low, high=high),
                                            (50, 50, 50), (30, 25, 20)),
                desc=f'{desc}_1d_target_lengths_intlists',
                reference_fn=reference_fn)
        )
        module_inputs.append(
            ModuleInput(
                constructor_input=FunctionInput(**constructor_kwargs),
                forward_input=FunctionInput(make_input((50, 3, 15)).log_softmax(2),
                                            make_target((30 + 25 + 20,), dtype=target_dtype, low=low, high=high),
                                            torch.tensor((50, 50, 50), device=device),
                                            torch.tensor((30, 25, 20), device=device)),
                desc=f'{desc}_1d_target_lengths_tensors',
                reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_GroupNorm(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(3, 6, 1e-3),
            forward_input=FunctionInput(make_input((4, 6, 5))),
            desc='1d_affine'),
        ModuleInput(
            constructor_input=FunctionInput(3, 12, 1e-3),
            forward_input=FunctionInput(make_input((4, 12))),
            desc='1d_affine_GN'),
        ModuleInput(
            constructor_input=FunctionInput(1, 6, 1e-3),
            forward_input=FunctionInput(make_input((150, 6))),
            desc='1d_affine_large_batch'),
        ModuleInput(
            constructor_input=FunctionInput(5, 5, 1e-3, False),
            forward_input=FunctionInput(make_input((4, 5, 5))),
            desc='1d_no_affine_IN'),
        ModuleInput(
            constructor_input=FunctionInput(1, 10, 1e-3, False),
            forward_input=FunctionInput(make_input((4, 10))),
            desc='1d_no_affine_LN'),
        ModuleInput(
            constructor_input=FunctionInput(3, 6, 1e-3),
            forward_input=FunctionInput(make_input((4, 6, 2, 3))),
            desc='2d_affine'),
        ModuleInput(
            constructor_input=FunctionInput(3, 3, 1e-3, False),
            forward_input=FunctionInput(make_input((4, 3, 2, 3))),
            desc='2d_no_affine_IN'),
        ModuleInput(
            constructor_input=FunctionInput(1, 3, 1e-3, False),
            forward_input=FunctionInput(make_input((4, 3, 2, 3))),
            desc='2d_no_affine_LN'),
    ]


def module_inputs_torch_nn_Hardshrink(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(2.),
            forward_input=FunctionInput(make_input((4, 3, 2, 4))),
        ),
        ModuleInput(
            constructor_input=FunctionInput(2.),
            forward_input=FunctionInput(make_input(())),
            desc='scalar',
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(4)),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim',
        )
    ]


def module_inputs_torch_nn_Hardswish(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(4)),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim',
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input((2, 3, 2, 5))),
            desc='4d_input')
    ]


def module_inputs_torch_nn_Hardtanh(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input((3, 2, 5))),
            reference_fn=lambda m, p, i: i.clamp(-1, 1),
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(())),
            reference_fn=lambda m, p, i: i.clamp(-1, 1),
            desc='scalar',
        ),
        ModuleInput(
            constructor_input=FunctionInput(),
            forward_input=FunctionInput(make_input(4)),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim',
        )
    ]


def module_inputs_torch_nn_HingeEmbeddingLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
        ('margin', {'margin': 0.5})
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return hingeembeddingloss_reference(i, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((10,)),
                                                    make_target((10,)).gt(0).to(dtype).mul_(2).sub_(1)),
                        desc=desc,
                        reference_fn=reference_fn)
        )
        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input(()),
                                                    make_target(()).gt(0).to(dtype).mul_(2).sub_(1)),
                        desc=f'scalar_{desc}',
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_HuberLoss(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases: list[tuple[str, dict]] = [
        ('', {}),
        ('reduction_sum', {'reduction': 'sum'}),
        ('reduction_mean', {'reduction': 'mean'}),
        ('reduction_none', {'reduction': 'none'}),
    ]

    module_inputs = []
    for desc, constructor_kwargs in cases:
        def reference_fn(m, p, i, t, constructor_kwargs=constructor_kwargs):
            return huberloss_reference(i, t, **constructor_kwargs)

        module_inputs.append(
            ModuleInput(constructor_input=FunctionInput(**constructor_kwargs),
                        forward_input=FunctionInput(make_input((5, 10)),
                                                    make_input((5, 10))),
                        desc=desc,
                        reference_fn=reference_fn)
        )

    return module_inputs


def module_inputs_torch_nn_InstanceNormNd(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    lazy = kwargs.get('lazy', False)
    N = kwargs['N']
    num_features, eps, momentum, affine, track_running_stats = 3, 1e-3, 0.3, False, True
    input_no_batch_shape_dict = {1: (3, 15), 2: (3, 6, 6), 3: (3, 4, 4, 4)}
    input_no_batch_shape = input_no_batch_shape_dict[N]
    input_batch_shape = (4,) + input_no_batch_shape

    return [
        ModuleInput(
            constructor_input=(
                FunctionInput(eps, momentum) if lazy else FunctionInput(num_features, eps, momentum)
            ),
            forward_input=FunctionInput(make_input(input_batch_shape))),
        ModuleInput(
            constructor_input=(
                FunctionInput(eps, momentum, affine, track_running_stats) if lazy else
                FunctionInput(num_features, eps, momentum, affine, track_running_stats)
            ),
            forward_input=FunctionInput(make_input(input_batch_shape)),
            desc='tracking_stats'),
        ModuleInput(
            constructor_input=(
                FunctionInput(eps, momentum) if lazy else FunctionInput(num_features, eps, momentum)
            ),
            forward_input=FunctionInput(make_input(input_no_batch_shape)),
            reference_fn=no_batch_dim_reference_fn,
            desc='tracking_stats_no_batch_dim'),
        ModuleInput(
            constructor_input=(
                FunctionInput(eps, momentum, affine, track_running_stats) if lazy else
                FunctionInput(num_features, eps, momentum, affine, track_running_stats)
            ),
            forward_input=FunctionInput(make_input(input_no_batch_shape)),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim')
    ]

def module_inputs_torch_nn_LayerNorm(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((4, 5, 5))),
            desc='1d_elementwise_affine'),
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((128, 5, 5))),
            desc='1d_elementwise_affine_large_batch'),
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3, False),
            forward_input=FunctionInput(make_input((4, 5, 5))),
            desc='1d_no_elementwise_affine'),
        ModuleInput(
            constructor_input=FunctionInput([2, 2, 5], 1e-3),
            forward_input=FunctionInput(make_input((4, 2, 2, 5))),
            desc='3d_elementwise_affine'),
        ModuleInput(
            constructor_input=FunctionInput([2, 2, 5], 1e-3, False),
            forward_input=FunctionInput(make_input((4, 2, 2, 5))),
            desc='3d_no_elementwise_affine'),
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((0, 5))),
            desc='1d_empty_elementwise_affine'),
        ModuleInput(
            constructor_input=FunctionInput([2, 2, 5], 1e-3, elementwise_affine=True, bias=False),
            forward_input=FunctionInput(make_input((4, 2, 2, 5))),
            desc='3d_elementwise_affine_no_bias'),
    ]

def module_inputs_torch_nn_RMSNorm(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def rms_norm_reference_fn(m, p, i):
        eps = m.eps
        if eps is None:
            eps = torch.finfo(i.dtype).eps
        ndim = i.ndim
        normalized_shape = m.normalized_shape
        weight = m.weight
        dims = [ndim - i - 1 for i in range(len(normalized_shape))]
        upcasted_i = i.float()
        result = upcasted_i * torch.rsqrt(upcasted_i.pow(2).mean(dim=dims, keepdim=True) + m.eps)
        if weight is not None:
            result *= weight
        return result.type_as(i)

    return [
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((4, 5, 5))),
            desc='1d_elementwise_affine',
            reference_fn=rms_norm_reference_fn),
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((128, 5, 5))),
            desc='1d_elementwise_affine_large_batch',
            reference_fn=rms_norm_reference_fn),
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3, False),
            forward_input=FunctionInput(make_input((4, 5, 5))),
            desc='1d_no_elementwise_affine',
            reference_fn=rms_norm_reference_fn),
        ModuleInput(
            constructor_input=FunctionInput([2, 2, 5], 1e-3),
            forward_input=FunctionInput(make_input((4, 2, 2, 5))),
            desc='3d_elementwise_affine',
            reference_fn=rms_norm_reference_fn),
        ModuleInput(
            constructor_input=FunctionInput([2, 2, 5], 1e-3, False),
            forward_input=FunctionInput(make_input((4, 2, 2, 5))),
            desc='3d_no_elementwise_affine',
            reference_fn=rms_norm_reference_fn),
        ModuleInput(
            constructor_input=FunctionInput([5], 1e-3),
            forward_input=FunctionInput(make_input((0, 5))),
            desc='1d_empty_elementwise_affine',
            reference_fn=rms_norm_reference_fn),
    ]


def module_inputs_torch_nn_LocalResponseNorm(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(3,),
            forward_input=FunctionInput(make_input((1, 5, 7))),
            desc='1d'),
        ModuleInput(
            constructor_input=FunctionInput(2,),
            forward_input=FunctionInput(make_input((1, 5, 7, 7))),
            desc='2d_uneven_pad'),
        ModuleInput(
            constructor_input=FunctionInput(1, 1., 0.5, 2.),
            forward_input=FunctionInput(make_input((1, 5, 7, 7, 7))),
            desc='3d_custom_params'),
    ]


def module_inputs_torch_nn_LPPool1d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(1.5, 2),
            forward_input=FunctionInput(make_input((1, 3, 7))),
            desc='norm'),
        ModuleInput(
            constructor_input=FunctionInput(2, 2, 3),
            forward_input=FunctionInput(make_input((1, 3, 7)))),
        ModuleInput(
            constructor_input=FunctionInput(2, 2, 3),
            forward_input=FunctionInput(make_input((3, 7))),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim'),
    ]



def module_inputs_torch_nn_LPPool2d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(2, 2, 2),
            forward_input=FunctionInput(make_input((1, 3, 7, 7)))),
        ModuleInput(
            constructor_input=FunctionInput(2, 2, 2),
            forward_input=FunctionInput(make_input((3, 7, 7))),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim'),
        ModuleInput(
            constructor_input=FunctionInput(1.5, 2),
            forward_input=FunctionInput(make_input((1, 3, 7, 7))),
            desc='norm'),
    ]


def module_inputs_torch_nn_LPPool3d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(2, 2, 2),
            forward_input=FunctionInput(make_input((1, 3, 7, 7, 7)))),
        ModuleInput(
            constructor_input=FunctionInput(2, 2, 2),
            forward_input=FunctionInput(make_input((3, 7, 7, 7))),
            reference_fn=no_batch_dim_reference_fn,
            desc='no_batch_dim'),
        ModuleInput(
            constructor_input=FunctionInput(1.5, 2),
            forward_input=FunctionInput(make_input((1, 3, 7, 7, 7))),
            desc='norm'),
    ]


def module_inputs_torch_nn_MaxPool1d(module_info, device, dtype, requires_grad, training, **kwargs):
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    return [
        ModuleInput(
            constructor_input=FunctionInput(4),
            forward_input=FunctionInput(make_input((2, 10, 4))),
            desc='3d_input'),
        ModuleInput(
            constructor_input=FunctionInput(4, 4),
            forward_input=FunctionInput(make_input((2, 10, 4))),
            desc='stride'),
        ModuleInput(
            constructor_input=FunctionInput(4, return_indices=True),
            forward_input=FunctionInput(make_input((2, 10, 4))),
      

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 6 class(es): modules, FunctionInput, ModuleInput, ModuleErrorEnum, ErrorModuleInput, ModuleInfo

### Functions
This file defines 155 function(s): __init__, _get_training_flags, _parametrize_test, test_wrapper, get_module_common_name, __init__, __init__, copy_reference_fn, __init__, __init__, get_decorators, supported_dtypes, name, formatted_name, module_inputs_torch_nn_Linear, module_inputs_torch_nn_Bilinear, bilinear_reference_fn, module_inputs_torch_nn_KLDivLoss, reference_fn, module_inputs_torch_nn_NLLLoss, make_input, reference_fn, nd_reference_fn, module_inputs_torch_nn_GaussianNLLLoss, module_inputs_torch_nn_PoissonNLLLoss, poissonnllloss_reference_fn, reference_fn, module_inputs_torch_nn_MSELoss, mse_loss_reference_fn, no_batch_dim_reference_fn


## Key Components

The file contains 11767 words across 4353 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 209310 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
